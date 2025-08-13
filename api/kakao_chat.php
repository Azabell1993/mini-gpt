<?php
// 내 카카오톡으로 메시지 전송 (POST: message, GET: uid)
header('Content-Type: application/json; charset=utf-8');
require_once __DIR__ . '/config.php';
require_once __DIR__ . '/inquiry_db.php';

// uid 필수
$userId = $_GET['uid'] ?? null;
if (!$userId) {
    http_response_code(400);
    echo json_encode(['error'=>'uid 파라미터 필요']);
    exit;
}

$tokenFile = __DIR__ . "/tokens/$userId.json";
if (!file_exists($tokenFile)) {
    http_response_code(404);
    echo json_encode(['error'=>'토큰 파일 없음']);
    exit;
}
$tok = json_decode(file_get_contents($tokenFile), true);
$access = $tok['access_token'] ?? null;
if (!$access) {
    http_response_code(401);
    echo json_encode(['error'=>'access_token 없음']);
    exit;
}


// 메시지 본문 (POST: name, email, phone, message)
$input = json_decode(file_get_contents('php://input'), true);
$name = isset($input['name']) ? trim($input['name']) : '';
$email = isset($input['email']) ? trim($input['email']) : '';
$phone = isset($input['phone']) ? trim($input['phone']) : '';
$msg = isset($input['message']) ? trim($input['message']) : '';
$ip = get_client_ip();

// 2분 딜레이 체크
if (!can_send_inquiry($ip)) {
    http_response_code(429);
    echo json_encode(['error' => '2분 내 연속으로 문의를 보낼 수 없습니다.']);
    exit;
}
if (!$name) {
    http_response_code(400);
    echo json_encode(['error' => '이름을 입력하세요.']);
    exit;
}
if (!$email && !$phone) {
    http_response_code(400);
    echo json_encode(['error' => '이메일 또는 핸드폰 번호 중 하나는 입력해야 합니다.']);
    exit;
}
if (!$msg) {
    http_response_code(400);
    echo json_encode(['error' => '문의 내용을 입력하세요.']);
    exit;
}
// 문의 DB 저장
save_inquiry($name, $email, $phone, $msg, $ip);

$info = [];
$info[] = "이름: $name";
if ($email) $info[] = "이메일: $email";
if ($phone) $info[] = "핸드폰: $phone";
$info[] = "문의: $msg";
$fullMsg = implode("\n", $info);

$payload = http_build_query([
  'template_object' => json_encode([
    'object_type' => 'text',
    'text'        => $fullMsg,
    'link'        => [
      'web_url' => 'https://your.domain',
      'mobile_web_url' => 'https://your.domain'
    ],
    'button_title' => '열기'
  ], JSON_UNESCAPED_UNICODE)
]);

$ch = curl_init('https://kapi.kakao.com/v2/api/talk/memo/default/send');
curl_setopt_array($ch, [
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_POST => true,
  CURLOPT_POSTFIELDS => $payload,
  CURLOPT_HTTPHEADER => [
    "Authorization: Bearer $access",
    "Content-Type: application/x-www-form-urlencoded"
  ]
]);
$res = curl_exec($ch);
$code = curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
$err = curl_error($ch);
curl_close($ch);

http_response_code($code);
if ($err) {
    echo json_encode(['error' => $err]);
    exit;
}

$json = json_decode($res, true);
if (isset($json['result_code']) && $json['result_code'] == 0) {
  echo json_encode(['result' => 'success', 'msg' => '문의가 완료되었습니다.']);
} else {
  echo $res;
}