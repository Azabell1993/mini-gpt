<?php
// 환경변수 로드 함수
function load_env($path) {
  if (!file_exists($path)) return;
  $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
  foreach ($lines as $line) {
    if (strpos(trim($line), '#') === 0) continue;
    if (!strpos($line, '=')) continue;
    list($name, $value) = array_map('trim', explode('=', $line, 2));
    putenv("$name=$value");
  }
}

// .env.local 파일 경로 (html 폴더보다 상위)
$env_path = dirname(__DIR__, 1) . '/.env.local';
load_env($env_path);

$OPENAI_API_KEY = getenv('OPENAI_API_KEY');
$KAKAO_APP_KEY = getenv('KAKAO_APP_KEY');
$KAKAO_ADMIN_KEY = getenv('KAKAO_ADMIN_KEY');
$KAKAO_REST_API_KEY = getenv('KAKAO_REST_API_KEY');
$KAKA_CLIENT_SECRET = getenv('KAKA_CLIENT_SECRET');
$KAKAO_REDIRECT_URI = getenv('KAKAO_REDIRECT_URI');
$KAKAO_REFRESH_TOKEN = getenv('KAKAO_REFRESH_TOKEN');
$KAKAO_AUTHORIZATION_CODE = getenv('KAKAO_AUTHORIZATION_CODE');
