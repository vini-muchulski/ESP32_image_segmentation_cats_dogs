#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "esp_heap_caps.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_flash.h"
#include "esp_timer.h"
#include "esp_littlefs.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const char *ssid = "Starlink";
const char *password = "diversao";
const int serverPort = 80;

constexpr const char *kModelName = "Gatos Cachorro UNet v5 Fast-SCNN 96px INT8";
constexpr const char *kModelPath = "/littlefs/gatos_v5_fast_scnn.tflite";
constexpr const char *kMountPath = "/littlefs";

struct ModelContext
{
  tflite::ErrorReporter *error_reporter;
  const tflite::Model *model;
  tflite::MicroInterpreter *interpreter;
  TfLiteTensor *input_tensor;
  TfLiteTensor *output_tensor;
  uint8_t *tensor_arena;
  uint8_t *model_buffer;
  size_t model_bytes;
  uint8_t *mask_buffer;
  size_t mask_buffer_size;
  bool initialized;

  static constexpr int kInputWidth = 96;
  static constexpr int kInputHeight = 96;
  static constexpr int kInputChannels = 3;
  static constexpr int kOutputClasses = 3;
  static constexpr int kImageSize = kInputWidth * kInputHeight * kInputChannels;
  static constexpr int kMaskSize = kInputWidth * kInputHeight;
  static constexpr int kTensorArenaSize = 4 * 1024 * 1024;
};

ModelContext model_ctx = {nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr, 0,       nullptr, 0, false};

struct InferenceResult
{
  bool success = false;
  char error_message[128] = {0};
  uint16_t mask_width = 0;
  uint16_t mask_height = 0;
  const uint8_t *mask_data = nullptr;
  size_t mask_size = 0;
  const uint8_t *logits_data = nullptr;
  size_t logits_size = 0;
  uint16_t logits_channels = 0;
  const char *logits_dtype = nullptr;
  uint32_t input_hash = 0;
  uint32_t logits_hash = 0;
  int64_t input_copy_us = 0;
  int64_t inference_us = 0;
  int64_t postprocess_us = 0;
  int64_t total_processing_us = 0;
  size_t internal_before = 0;
  size_t internal_after = 0;
  size_t psram_before = 0;
  size_t psram_after = 0;
  size_t min_free_internal = 0;
  size_t min_free_psram = 0;
};

static uint32_t fnv1a_32(const void *data, size_t size)
{
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < size; ++i)
  {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static void set_error(InferenceResult &result, const char *msg)
{
  snprintf(result.error_message, sizeof(result.error_message), "%s", msg);
  result.success = false;
}

void cleanup_model();
bool connect_wifi();
bool initialize_model_ctx();
InferenceResult run_inference(const uint8_t *image_data);
void print_mem_telemetry();
void print_model_memory_report(size_t internal_before,
                               size_t internal_after,
                               size_t psram_before,
                               size_t psram_after);

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1
#define MAXIMUM_RETRY 30

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
  {
    esp_wifi_connect();
  }
  else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
  {
    printf("WiFi desconectado - tentando reconectar...\n");
    esp_wifi_connect();
  }
  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
  {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    printf("WiFi conectado! IP: " IPSTR " Porta: %d\n",
           IP2STR(&event->ip_info.ip), serverPort);
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

bool connect_wifi()
{
  printf("=== Conectando ao WiFi ===\n");
  printf("SSID: %s\n", ssid);

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  esp_event_handler_instance_t instance_any_id;
  esp_event_handler_instance_t instance_got_ip;
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL,
      &instance_any_id));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL,
      &instance_got_ip));

  wifi_config_t wifi_config = {};
  strncpy((char *)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid));
  strncpy((char *)wifi_config.sta.password, password,
          sizeof(wifi_config.sta.password));
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());

  EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                         WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                         pdFALSE, pdFALSE,
                                         pdMS_TO_TICKS(30000));
  if (bits & WIFI_CONNECTED_BIT)
  {
    return true;
  }
  printf("Falha na conexão WiFi (timeout)\n");
  return false;
}

void cleanup_model()
{
  if (model_ctx.model_buffer)
  {
    free(model_ctx.model_buffer);
    model_ctx.model_buffer = nullptr;
  }
  if (model_ctx.mask_buffer)
  {
    free(model_ctx.mask_buffer);
    model_ctx.mask_buffer = nullptr;
    model_ctx.mask_buffer_size = 0;
  }
  if (model_ctx.tensor_arena)
  {
    free(model_ctx.tensor_arena);
    model_ctx.tensor_arena = nullptr;
  }
  model_ctx.initialized = false;
}

void *allocate_memory(size_t size)
{
  void *ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (ptr == nullptr)
  {
    ptr = malloc(size);
  }
  return ptr;
}

void print_mem_telemetry()
{
  printf("Free heap (geral): %u bytes\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)));
  printf("Free PSRAM: %u bytes\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  printf("Free INTERNAL: %u bytes\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
  printf("Largest INTERNAL block: %u bytes\n",
         static_cast<unsigned>(
             heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)));
}

void print_model_memory_report(size_t internal_before,
                               size_t internal_after,
                               size_t psram_before,
                               size_t psram_after)
{
  uint32_t flash_bytes = 0;
  esp_flash_get_size(NULL, &flash_bytes);
  const esp_partition_t *running_partition = esp_ota_get_running_partition();
  const size_t app_partition_bytes =
      running_partition ? running_partition->size : 0;
  const size_t arena_used = model_ctx.interpreter->arena_used_bytes();
  const size_t arena_allocated =
      heap_caps_get_allocated_size(model_ctx.tensor_arena);

  printf("\n=== Relatório de memória do modelo ===\n");
  printf("Modelo TFLite no littlefs: %u bytes (%.2f MiB)\n",
         static_cast<unsigned>(model_ctx.model_bytes),
         model_ctx.model_bytes / (1024.0f * 1024.0f));
  printf("Cópia do modelo em RAM/PSRAM: %u bytes\n",
         static_cast<unsigned>(model_ctx.model_bytes));
  printf("Partição da aplicação: %u bytes\n",
         static_cast<unsigned>(app_partition_bytes));
  if (app_partition_bytes > 0)
  {
    printf("Modelo ocupa %.2f%% da partição da aplicação\n",
           100.0f * model_ctx.model_bytes / app_partition_bytes);
  }
  printf("Flash física do ESP32: %u bytes\n",
         static_cast<unsigned>(flash_bytes));

  printf("Tensor arena usada: %u bytes\n",
         static_cast<unsigned>(arena_used));
  printf("Tensor arena alocada: %u bytes | limite configurado: %u bytes\n",
         static_cast<unsigned>(arena_allocated),
         static_cast<unsigned>(ModelContext::kTensorArenaSize));
  printf("Memória interna consumida na inicialização: %d bytes\n",
         static_cast<int>(internal_before) -
             static_cast<int>(internal_after));
  printf("PSRAM consumida na inicialização: %d bytes\n",
         static_cast<int>(psram_before) -
             static_cast<int>(psram_after));
  printf("Buffer HTTP temporário por imagem: %u bytes\n",
         static_cast<unsigned>(ModelContext::kImageSize));
  printf("Buffer persistente da máscara após a primeira inferência: %u bytes\n",
         static_cast<unsigned>(ModelContext::kMaskSize));
  printf("=== Fim do relatório de memória ===\n");
}

bool load_model()
{
  printf("[1] Carregando modelo do littlefs...\n");
  FILE *f = fopen(kModelPath, "rb");
  if (!f)
  {
    printf("ERRO: Falha ao abrir %s (rode 'pio run -t uploadfs')\n",
           kModelPath);
    return false;
  }

  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0)
  {
    fclose(f);
    printf("ERRO: Modelo vazio\n");
    return false;
  }

  model_ctx.model_buffer = static_cast<uint8_t *>(allocate_memory(sz));
  if (!model_ctx.model_buffer)
  {
    fclose(f);
    printf("ERRO: Falha ao alocar %ld bytes para o modelo\n", sz);
    return false;
  }

  size_t rd = fread(model_ctx.model_buffer, 1, sz, f);
  fclose(f);
  if (rd != (size_t)sz)
  {
    printf("ERRO: Leitura incompleta (%u/%ld)\n",
           static_cast<unsigned>(rd), sz);
    return false;
  }
  model_ctx.model_bytes = static_cast<size_t>(sz);

  model_ctx.model = tflite::GetModel(model_ctx.model_buffer);
  if (model_ctx.model == nullptr)
  {
    printf("ERRO: Falha ao carregar modelo\n");
    return false;
  }

  if (model_ctx.model->version() != TFLITE_SCHEMA_VERSION)
  {
    printf("ERRO: Versão incompatível: %u vs %u\n",
           static_cast<unsigned>(model_ctx.model->version()),
           TFLITE_SCHEMA_VERSION);
    return false;
  }

  printf("Modelo carregado com sucesso (%ld bytes)\n", sz);
  return true;
}

bool initialize_interpreter()
{
  printf("[2] Inicializando interpretador...\n");

  model_ctx.tensor_arena = static_cast<uint8_t *>(
      allocate_memory(ModelContext::kTensorArenaSize));

  if (model_ctx.tensor_arena == nullptr)
  {
    printf("ERRO: Falha na alocação de %d bytes\n",
           ModelContext::kTensorArenaSize);
    return false;
  }

  static tflite::MicroMutableOpResolver<9> op_resolver;
  op_resolver.AddAdd();
  op_resolver.AddAveragePool2D();
  op_resolver.AddBatchToSpaceNd();
  op_resolver.AddConcatenation();
  op_resolver.AddConv2D();
  op_resolver.AddDepthwiseConv2D();
  op_resolver.AddMul();
  op_resolver.AddResizeBilinear();
  op_resolver.AddSpaceToBatchNd();

  static tflite::MicroInterpreter static_interpreter(
      model_ctx.model, op_resolver, model_ctx.tensor_arena,
      ModelContext::kTensorArenaSize);
  model_ctx.interpreter = &static_interpreter;

  TfLiteStatus allocate_status = model_ctx.interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    printf("ERRO: AllocateTensors falhou (código: %d)\n", allocate_status);
    return false;
  }

  model_ctx.input_tensor = model_ctx.interpreter->input(0);
  model_ctx.output_tensor = model_ctx.interpreter->output(0);

  if (model_ctx.input_tensor == nullptr || model_ctx.output_tensor == nullptr)
  {
    printf("ERRO: Ponteiros de tensor nulos\n");
    return false;
  }

  if (model_ctx.input_tensor->dims->size != 4)
  {
    printf("ERRO: Tensor de entrada não é NHWC\n");
    return false;
  }
  const int in_h = model_ctx.input_tensor->dims->data[1];
  const int in_w = model_ctx.input_tensor->dims->data[2];
  const int in_c = model_ctx.input_tensor->dims->data[3];
  if (model_ctx.input_tensor->type != kTfLiteInt8 &&
      model_ctx.input_tensor->type != kTfLiteUInt8)
  {
    printf("ERRO: Tipo de entrada não suportado\n");
    return false;
  }
  if (in_h != ModelContext::kInputHeight ||
      in_w != ModelContext::kInputWidth ||
      in_c != ModelContext::kInputChannels)
  {
    printf("ERRO: Shape de entrada inesperada: %dx%dx%d (esperado %dx%dx%d)\n",
           in_w, in_h, in_c,
           ModelContext::kInputWidth, ModelContext::kInputHeight,
           ModelContext::kInputChannels);
    return false;
  }
  if (model_ctx.input_tensor->bytes != ModelContext::kImageSize)
  {
    printf("ERRO: Tamanho do tensor de entrada inesperado\n");
    return false;
  }

  const char *input_type_name =
      model_ctx.input_tensor->type == kTfLiteInt8 ? "int8" : "uint8";
  printf("Entrada: %dx%dx%d type=%s scale=%.9f zero_point=%ld\n",
         in_w, in_h, in_c, input_type_name,
         model_ctx.input_tensor->params.scale,
         (long)model_ctx.input_tensor->params.zero_point);
  printf("Arena usada: %u/%d bytes\n",
         static_cast<unsigned>(model_ctx.interpreter->arena_used_bytes()),
         ModelContext::kTensorArenaSize);
  printf("Interpretador inicializado com sucesso\n");
  return true;
}

bool initialize_model_ctx()
{
  printf("=== Inicializando Modelo %s ===\n", kModelName);

  static tflite::MicroErrorReporter micro_error_reporter;
  model_ctx.error_reporter = &micro_error_reporter;

  if (!load_model())
  {
    return false;
  }

  if (!initialize_interpreter())
  {
    cleanup_model();
    return false;
  }

  model_ctx.initialized = true;
  printf("=== Modelo inicializado com sucesso ===\n");
  return true;
}

void load_input_quantized(const uint8_t *src)
{
  constexpr float kNormMean[3] = {0.485f, 0.456f, 0.406f};
  constexpr float kNormStd[3] = {0.229f, 0.224f, 0.225f};
  const float scale = model_ctx.input_tensor->params.scale;
  const int32_t zero_point = model_ctx.input_tensor->params.zero_point;

  if (model_ctx.input_tensor->type == kTfLiteInt8)
  {
    int8_t *dst = model_ctx.input_tensor->data.int8;
    for (int i = 0; i < ModelContext::kImageSize; ++i)
    {
      const int c = i % ModelContext::kInputChannels;
      const float v = (static_cast<float>(src[i]) / 255.0f - kNormMean[c]) /
                      kNormStd[c];
      int32_t q = static_cast<int32_t>(lrintf(v / scale)) + zero_point;
      q = std::min<int32_t>(127, std::max<int32_t>(-128, q));
      dst[i] = static_cast<int8_t>(q);
    }
    return;
  }

  if (model_ctx.input_tensor->type == kTfLiteUInt8)
  {
    uint8_t *dst = model_ctx.input_tensor->data.uint8;
    for (int i = 0; i < ModelContext::kImageSize; ++i)
    {
      const int c = i % ModelContext::kInputChannels;
      const float v = (static_cast<float>(src[i]) / 255.0f - kNormMean[c]) /
                      kNormStd[c];
      int32_t q = static_cast<int32_t>(lrintf(v / scale)) + zero_point;
      q = std::min<int32_t>(255, std::max<int32_t>(0, q));
      dst[i] = static_cast<uint8_t>(q);
    }
  }
}

InferenceResult run_inference(const uint8_t *image_data)
{
  InferenceResult result{};
  if (!model_ctx.initialized)
  {
    set_error(result, "Modelo não inicializado");
    return result;
  }

  result.internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  result.psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  const int64_t total_start = esp_timer_get_time();

  const int64_t input_start = esp_timer_get_time();
  load_input_quantized(image_data);
  result.input_hash = fnv1a_32(model_ctx.input_tensor->data.raw,
                              model_ctx.input_tensor->bytes);
  result.input_copy_us = esp_timer_get_time() - input_start;

  const int64_t inference_start = esp_timer_get_time();
  const TfLiteStatus invoke_status = model_ctx.interpreter->Invoke();
  result.inference_us = esp_timer_get_time() - inference_start;

  if (invoke_status != kTfLiteOk)
  {
    set_error(result, "Falha na inferência");
    result.internal_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    result.psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    return result;
  }

  const int64_t postprocess_start = esp_timer_get_time();
  if (model_ctx.output_tensor->dims->size != 4)
  {
    set_error(result, "Tensor de saída não é NHWC");
    return result;
  }
  if (model_ctx.output_tensor->type != kTfLiteInt8 &&
      model_ctx.output_tensor->type != kTfLiteUInt8)
  {
    set_error(result, "Tipo de saída não suportado");
    return result;
  }
  const int out_h = model_ctx.output_tensor->dims->data[1];
  const int out_w = model_ctx.output_tensor->dims->data[2];
  const int out_c = model_ctx.output_tensor->dims->data[3];
  if (out_c != ModelContext::kOutputClasses)
  {
    set_error(result, "Saída com canais inesperados");
    return result;
  }
  const size_t mask_size = static_cast<size_t>(out_h) *
                           static_cast<size_t>(out_w);
  if (mask_size == 0)
  {
    set_error(result, "Saída vazia");
    return result;
  }

  result.logits_data = reinterpret_cast<const uint8_t *>(
      model_ctx.output_tensor->data.raw);
  result.logits_size = mask_size * static_cast<size_t>(out_c);
  result.logits_channels = static_cast<uint16_t>(out_c);
  result.logits_dtype = model_ctx.output_tensor->type == kTfLiteInt8
                            ? "int8"
                            : "uint8";
  result.logits_hash = fnv1a_32(result.logits_data, result.logits_size);

  if (model_ctx.mask_buffer_size < mask_size)
  {
    if (model_ctx.mask_buffer)
    {
      free(model_ctx.mask_buffer);
      model_ctx.mask_buffer = nullptr;
      model_ctx.mask_buffer_size = 0;
    }
    model_ctx.mask_buffer =
        static_cast<uint8_t *>(allocate_memory(mask_size));
    if (!model_ctx.mask_buffer)
    {
      set_error(result, "Falha ao alocar buffer de máscara");
      return result;
    }
    model_ctx.mask_buffer_size = mask_size;
  }

  if (model_ctx.output_tensor->type == kTfLiteInt8)
  {
    const int8_t *out = model_ctx.output_tensor->data.int8;
    for (size_t i = 0; i < mask_size; ++i)
    {
      int best = 0;
      int8_t best_v = out[i * out_c];
      for (int c = 1; c < out_c; ++c)
      {
        const int8_t v = out[i * out_c + c];
        if (v > best_v)
        {
          best_v = v;
          best = c;
        }
      }
      model_ctx.mask_buffer[i] = static_cast<uint8_t>(best);
    }
  }
  else
  {
    const uint8_t *out = model_ctx.output_tensor->data.uint8;
    for (size_t i = 0; i < mask_size; ++i)
    {
      int best = 0;
      uint8_t best_v = out[i * out_c];
      for (int c = 1; c < out_c; ++c)
      {
        const uint8_t v = out[i * out_c + c];
        if (v > best_v)
        {
          best_v = v;
          best = c;
        }
      }
      model_ctx.mask_buffer[i] = static_cast<uint8_t>(best);
    }
  }

  result.success = true;
  result.mask_width = static_cast<uint16_t>(out_w);
  result.mask_height = static_cast<uint16_t>(out_h);
  result.mask_data = model_ctx.mask_buffer;
  result.mask_size = mask_size;
  result.postprocess_us = esp_timer_get_time() - postprocess_start;
  result.total_processing_us = esp_timer_get_time() - total_start;
  result.internal_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  result.psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  result.min_free_internal =
      heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
  result.min_free_psram = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
  return result;
}

static void build_status_json(char *buf, size_t len)
{
  snprintf(buf, len,
           "{\n"
           "  \"success\": %s,\n"
           "  \"mask_width\": %d,\n"
           "  \"mask_height\": %d,\n"
           "  \"mask_bytes\": %u,\n"
           "  \"error_message\": \"%s\",\n"
           "  \"heap_free\": %u,\n"
           "  \"model_initialized\": %s\n"
           "}",
           model_ctx.initialized ? "true" : "false",
           ModelContext::kInputWidth,
           ModelContext::kInputHeight,
           static_cast<unsigned>(ModelContext::kMaskSize),
           model_ctx.initialized ? "" : "Modelo não inicializado",
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
           model_ctx.initialized ? "true" : "false");
}

static void build_error_json(const InferenceResult &res, char *buf, size_t len)
{
  snprintf(buf, len,
           "{\n"
           "  \"success\": false,\n"
           "  \"mask_width\": 0,\n"
           "  \"mask_height\": 0,\n"
           "  \"mask_bytes\": 0,\n"
           "  \"error_message\": \"%s\",\n"
           "  \"heap_free\": %u,\n"
           "  \"model_initialized\": %s\n"
           "}",
           res.error_message,
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
           model_ctx.initialized ? "true" : "false");
}

static esp_err_t status_handler(httpd_req_t *req)
{
  char body[512];
  build_status_json(body, sizeof(body));
  httpd_resp_set_type(req, "application/json");
  httpd_resp_sendstr(req, body);
  return ESP_OK;
}

static esp_err_t root_handler(httpd_req_t *req)
{
  char body[512];
  snprintf(body, sizeof(body),
           "<!DOCTYPE html><html><body><h1>%s API</h1>"
           "<p><b>POST /predict_bin</b> %d B raw</p>"
           "<p>Retorna máscara de classes 0..%d (%dx%d)</p>"
           "<p><b>GET /status</b></p></body></html>",
           kModelName, ModelContext::kImageSize,
           ModelContext::kOutputClasses - 1,
           ModelContext::kInputWidth, ModelContext::kInputHeight);
  httpd_resp_set_type(req, "text/html");
  httpd_resp_sendstr(req, body);
  return ESP_OK;
}

static esp_err_t predict_bin_handler(httpd_req_t *req)
{
  InferenceResult res{};
  const bool return_logits =
      httpd_req_get_hdr_value_len(req, "X-Return-Logits") > 0;

  if (req->content_len != ModelContext::kImageSize)
  {
    set_error(res, "Content-Length inválido");
    char body[512];
    build_error_json(res, body, sizeof(body));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, body);
    return ESP_OK;
  }

  uint8_t *img = static_cast<uint8_t *>(
      allocate_memory(ModelContext::kImageSize));
  if (!img)
  {
    set_error(res, "Falha de memória");
    char body[512];
    build_error_json(res, body, sizeof(body));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, body);
    return ESP_OK;
  }

  const int64_t t_rx0 = esp_timer_get_time();
  int remaining = req->content_len;
  size_t off = 0;
  bool rx_ok = true;
  while (remaining > 0)
  {
    int r = httpd_req_recv(req, reinterpret_cast<char *>(img) + off,
                           static_cast<size_t>(remaining));
    if (r == HTTPD_SOCK_ERR_TIMEOUT)
    {
      continue;
    }
    if (r <= 0)
    {
      rx_ok = false;
      break;
    }
    off += static_cast<size_t>(r);
    remaining -= r;
  }
  const int64_t rx_us = esp_timer_get_time() - t_rx0;

  if (!rx_ok)
  {
    set_error(res, "Falha na recepção da imagem");
    char body[512];
    build_error_json(res, body, sizeof(body));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, body);
    free(img);
    return ESP_OK;
  }

  printf("RX ok em %lld us\n", static_cast<long long>(rx_us));
  printf("Mem pré-inferência | heap: %u | psram: %u | internal: %u\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));

  const int64_t t_inf0 = esp_timer_get_time();
  res = run_inference(img);
  printf("Processamento completo em %lld ms\n",
         static_cast<long long>((esp_timer_get_time() - t_inf0) / 1000));
  printf("Tempos | cópia: %lldus | inferência: %.6f s | pós: %lldus | total: %lldus\n",
         static_cast<long long>(res.input_copy_us),
         static_cast<double>(res.inference_us) / 1000000.0,
         static_cast<long long>(res.postprocess_us),
         static_cast<long long>(res.total_processing_us));
  printf("Hashes FNV-1a | input_int8: %08x | logits: %08x\n",
         static_cast<unsigned>(res.input_hash),
         static_cast<unsigned>(res.logits_hash));
  printf("Mem pós-inferência | heap: %u | psram: %u | internal: %u\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));

  free(img);

  if (!res.success)
  {
    char body[512];
    build_error_json(res, body, sizeof(body));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, body);
    return ESP_OK;
  }

  uint32_t flash_bytes = 0;
  esp_flash_get_size(NULL, &flash_bytes);
  const esp_partition_t *running_partition = esp_ota_get_running_partition();
  const size_t app_partition_bytes =
      running_partition ? running_partition->size : 0;

  char hdr[24][24];
  int hn = 0;
  httpd_resp_set_type(req, "application/octet-stream");
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u", res.mask_width);
  httpd_resp_set_hdr(req, "X-Width", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u", res.mask_height);
  httpd_resp_set_hdr(req, "X-Height", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u", res.logits_channels);
  httpd_resp_set_hdr(req, "X-Channels", hdr[hn++]);
  httpd_resp_set_hdr(req, "X-Response-Kind",
                     return_logits ? "logits" : "mask");
  httpd_resp_set_hdr(req, "X-Logits-Dtype", res.logits_dtype);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%08x",
           static_cast<unsigned>(res.input_hash));
  httpd_resp_set_hdr(req, "X-Input-FNV1A", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%08x",
           static_cast<unsigned>(res.logits_hash));
  httpd_resp_set_hdr(req, "X-Logits-FNV1A", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%lld",
           static_cast<long long>(res.input_copy_us));
  httpd_resp_set_hdr(req, "X-Input-Copy-Us", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%lld",
           static_cast<long long>(res.inference_us));
  httpd_resp_set_hdr(req, "X-Inference-Us", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%lld",
           static_cast<long long>(res.postprocess_us));
  httpd_resp_set_hdr(req, "X-Postprocess-Us", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%lld",
           static_cast<long long>(res.total_processing_us));
  httpd_resp_set_hdr(req, "X-Total-Processing-Us", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(res.internal_before));
  httpd_resp_set_hdr(req, "X-Internal-Before", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(res.internal_after));
  httpd_resp_set_hdr(req, "X-Internal-After", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(res.psram_before));
  httpd_resp_set_hdr(req, "X-Psram-Before", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(res.psram_after));
  httpd_resp_set_hdr(req, "X-Psram-After", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(res.min_free_internal));
  httpd_resp_set_hdr(req, "X-Min-Free-Internal", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(res.min_free_psram));
  httpd_resp_set_hdr(req, "X-Min-Free-Psram", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(model_ctx.interpreter->arena_used_bytes()));
  httpd_resp_set_hdr(req, "X-Arena-Used", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(ModelContext::kTensorArenaSize));
  httpd_resp_set_hdr(req, "X-Arena-Capacity", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(model_ctx.model_bytes));
  httpd_resp_set_hdr(req, "X-Model-Bytes", hdr[hn++]);
  httpd_resp_set_hdr(req, "X-Firmware-Bytes", "0");
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u",
           static_cast<unsigned>(app_partition_bytes));
  httpd_resp_set_hdr(req, "X-App-Partition-Bytes", hdr[hn++]);
  snprintf(hdr[hn], sizeof(hdr[hn]), "%u", static_cast<unsigned>(flash_bytes));
  httpd_resp_set_hdr(req, "X-Flash-Chip-Bytes", hdr[hn++]);

  const uint8_t *response_data =
      return_logits ? res.logits_data : res.mask_data;
  const size_t response_size =
      return_logits ? res.logits_size : res.mask_size;
  httpd_resp_send(req, reinterpret_cast<const char *>(response_data),
                  static_cast<int>(response_size));
  return ESP_OK;
}

static void start_server()
{
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = serverPort;
  cfg.stack_size = 16384;
  cfg.max_uri_handlers = 8;
  cfg.max_resp_headers = 32;

  httpd_handle_t server = NULL;
  if (httpd_start(&server, &cfg) != ESP_OK)
  {
    printf("ERRO: Falha ao iniciar servidor HTTP\n");
    return;
  }

  httpd_uri_t uri_predict = {
      .uri = "/predict_bin",
      .method = HTTP_POST,
      .handler = predict_bin_handler,
      .user_ctx = NULL};
  httpd_uri_t uri_status = {
      .uri = "/status",
      .method = HTTP_GET,
      .handler = status_handler,
      .user_ctx = NULL};
  httpd_uri_t uri_root = {
      .uri = "/",
      .method = HTTP_GET,
      .handler = root_handler,
      .user_ctx = NULL};
  httpd_register_uri_handler(server, &uri_predict);
  httpd_register_uri_handler(server, &uri_status);
  httpd_register_uri_handler(server, &uri_root);

  printf("=== Servidor HTTP iniciado ===\n");
}

static void init_littlefs()
{
  printf("=== Montando LittleFS ===\n");
  esp_vfs_littlefs_conf_t conf = {
      .base_path = kMountPath,
      .partition_label = "littlefs",
      .partition = NULL,
      .format_if_mount_failed = true,
      .read_only = false,
      .dont_mount = false,
      .grow_on_mount = false,
  };

  esp_err_t ret = esp_vfs_littlefs_register(&conf);
  if (ret != ESP_OK)
  {
    printf("ERRO: Falha ao montar littlefs (%d)\n", ret);
    return;
  }

  size_t total = 0;
  size_t used = 0;
  esp_littlefs_info(conf.partition_label, &total, &used);
  printf("LittleFS montado: %u bytes total, %u bytes usados\n",
         static_cast<unsigned>(total),
         static_cast<unsigned>(used));
}

extern "C" void app_main()
{
  esp_err_t nvs_ret = nvs_flash_init();
  if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
  {
    ESP_ERROR_CHECK(nvs_flash_erase());
    nvs_ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(nvs_ret);

  s_wifi_event_group = xEventGroupCreate();

  printf("\n=== %s TensorFlow Lite WiFi API (ESP-IDF) ===\n", kModelName);
  printf("Free heap inicial: %u bytes\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)));
  printf("PSRAM disponível: %u bytes\n",
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  print_mem_telemetry();

  init_littlefs();

  connect_wifi();

  const size_t internal_before_model =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const size_t psram_before_model =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

  if (!initialize_model_ctx())
  {
    printf("Falha na inicialização do modelo! Parando.\n");
    while (true)
    {
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }

  const size_t internal_after_model =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const size_t psram_after_model =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  print_model_memory_report(internal_before_model,
                            internal_after_model,
                            psram_before_model,
                            psram_after_model);

  start_server();
}
