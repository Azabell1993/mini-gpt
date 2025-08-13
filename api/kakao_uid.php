<?php
require_once __DIR__ . '/config.php';
header('Content-Type: application/json; charset=utf-8');
$uid = getenv('KAKAO_REST_API_KEY'); // REST API 키만 반환
if ($uid) {
    echo json_encode(['uid' => $uid]);
} else {
    http_response_code(404);
    echo json_encode(['error' => 'uid not found']);
}