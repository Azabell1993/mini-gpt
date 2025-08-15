<?php
// /api/send_inquiry.php
// PHP 8+ 권장. display_errors off.
// Apache/Nginx에서 JSON POST만 허용 권장.

header('Content-Type: application/json; charset=UTF-8');
header('X-Content-Type-Options: nosniff');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
  http_response_code(405);
  echo json_encode(['ok'=>false,'error'=>'METHOD_NOT_ALLOWED']);
  exit;
}

// .env.local 에서 설정 로드 (키=값)
$env = @parse_ini_file(__DIR__ . '/../.env.local', false, INI_SCANNER_TYPED) ?: [];
$SLACK_BOT_TOKEN = $env['SLACK_BOT_TOKEN'] ?? '';
$SLACK_CHANNEL_ID = $env['SLACK_CHANNEL_ID'] ?? '';

if (!$SLACK_BOT_TOKEN || !$SLACK_CHANNEL_ID) {
  http_response_code(500);
  echo json_encode(['ok'=>false,'error'=>'MISSING_SLACK_CONFIG']);
  exit;
}

$raw = file_get_contents('php://input');
$in  = json_decode($raw, true);
if (!is_array($in)) {
  http_response_code(400);
  echo json_encode(['ok'=>false,'error'=>'INVALID_JSON']);
  exit;
}

// 입력값 검증 (최소 수준)
$name    = trim($in['name']   ?? '');
$email   = trim($in['email']  ?? '');
$phone   = trim($in['phone']  ?? '');
$message = trim($in['message']?? '');

if ($name === '' || $email === '' || $message === '') {
  http_response_code(400);
  echo json_encode(['ok'=>false,'error'=>'REQUIRED_FIELDS']);
  exit;
}

// 사용자 메타 (IP는 수집하지 않음)
$ua  = $_SERVER['HTTP_USER_AGENT'] ?? '';
$ref = $_SERVER['HTTP_REFERER'] ?? '';

// Slack 메시지 구성 (Block Kit)
$blocks = [
  ['type'=>'header','text'=>['type'=>'plain_text','text'=>'💬 새 문의 도착']],
  ['type'=>'section','fields'=>[
    ['type'=>'mrkdwn','text'=>"*이름:*\n{$name}"],
    ['type'=>'mrkdwn','text'=>"*이메일:*\n{$email}"],
    ['type'=>'mrkdwn','text'=>"*핸드폰:*\n".($phone ?: '미제공')],
  ]],
  ['type'=>'section','text'=>['type'=>'mrkdwn','text'=>"*문의 내용:*\n".($message)]],
  ['type'=>'context','elements'=>[
    ['type'=>'mrkdwn','text'=>"UA: " . substr($ua,0,200)],
    ['type'=>'mrkdwn','text'=>"Referer: " . ($ref ?: 'N/A')],
  ]],
];

$payload = [
  'channel' => $SLACK_CHANNEL_ID,
  'text'    => "새 문의: {$name} - {$email}", // Fallback
  'blocks'  => $blocks,
];

// Slack API 호출
$ch = curl_init('https://slack.com/api/chat.postMessage');
curl_setopt_array($ch, [
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_POST           => true,
  CURLOPT_HTTPHEADER     => [
    'Content-Type: application/json; charset=utf-8',
    'Authorization: Bearer ' . $SLACK_BOT_TOKEN
  ],
  CURLOPT_POSTFIELDS     => json_encode($payload, JSON_UNESCAPED_UNICODE),
  CURLOPT_TIMEOUT        => 15,
]);
$resp = curl_exec($ch);
$errno = curl_errno($ch);
$code  = curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
curl_close($ch);

if ($errno) {
  http_response_code(502);
  echo json_encode(['ok'=>false,'error'=>'CURL_ERROR_'.$errno]);
  exit;
}

$data = json_decode($resp, true);
if (!$data || !($data['ok'] ?? false)) {
  http_response_code(502);
  echo json_encode(['ok'=>false,'error'=>'SLACK_API_FAIL','detail'=>$data]);
  exit;
}

echo json_encode(['ok'=>true]);