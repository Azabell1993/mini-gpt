<?php
header('Content-Type: application/json');
echo json_encode([
  "sample_logits" => [-0.186212, 0.162534, -0.100814, 0.409218, 0.045253, 0.224676, 0.173594, -0.243189, 0.007037, 0.419056],
  "inference_time_us" => 12505,
  "performance" => [
    "layer_0_time_us" => 32397.0,
    "layer_1_time_us" => 4821.0,
    "total_forward_time_us" => 12497.0,
    "tokens_per_second" => 400.096008
  ],
  "use_cache" => true,
  "output_shape" => [5, 1000],
  "input_tokens" => [2, 3, 22, 30, 31],
  "input_text" => "hello world from flash attention"
]);561 bytes transferred
lftp azabellcode@112.175.185.148:/html/api> 
lftp azabellcode@112.175.185.148:/html/api> cat generate.php 
<?php
header('Content-Type: application/json');
echo json_encode([
  "generated" => "Hello World!",
  "input_tokens" => [2, 3, 22, 30, 31],
  "output_tokens" => [999, 888, 777, 666, 555],
  "log_probs" => [-1.23, -2.34, -3.45, -4.56, -5.67],
  "token_count" => [
    "prompt_tokens" => 5,
    "generated_tokens" => 5,
    "total_tokens" => 10
  ],
  "timing" => [
    "total_time_ms" => 12.3,
    "tokens_per_second" => 400.1
  ]
]);