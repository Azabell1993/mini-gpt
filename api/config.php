<?php
header('Content-Type: application/json');
echo json_encode([
  "pre_ln" => true,
  "d_ff" => 512,
  "d_model" => 128,
  "dropout" => 0.1,
  "max_seq_len" => 64,
  "n_heads" => 4,
  "n_layers" => 2,
  "features" => ["flash_attention", "kv_cache", "advanced_sampling"],
  "vocab_size" => 1000
]);