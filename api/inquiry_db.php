<?php
// 문의 DB 저장 및 2분 딜레이 체크
function get_db() {
    $conf = require __DIR__ . '/mysql.php';
    $mysqli = new mysqli($conf['host'], $conf['user'], $conf['pass'], $conf['db']);
    if ($mysqli->connect_errno) die(json_encode(['error'=>'DB 연결 실패']));
    $mysqli->set_charset('utf8mb4');
    return $mysqli;
}

function get_client_ip() {
    // 공인 IP를 curl ifconfig.me로 가져옴
    $ip = @file_get_contents('http://ifconfig.me/ip');
    if ($ip && preg_match('/^\d+\.\d+\.\d+\.\d+$/', trim($ip))) {
        return trim($ip);
    }
    return $_SERVER['REMOTE_ADDR'] ?? 'unknown';
}

function can_send_inquiry($ip) {
    $db = get_db();
    $stmt = $db->prepare('SELECT created_at FROM inquiries WHERE ip=? ORDER BY created_at DESC LIMIT 1');
    $stmt->bind_param('s', $ip);
    $stmt->execute();
    $stmt->bind_result($created_at);
    if ($stmt->fetch()) {
        $last = strtotime($created_at);
        if (time() - $last < 120) return false;
    }
    $stmt->close();
    $db->close();
    return true;
}

function save_inquiry($name, $email, $phone, $message, $ip) {
    $db = get_db();
    $stmt = $db->prepare('INSERT INTO inquiries (name, email, phone, message, ip) VALUES (?, ?, ?, ?, ?)');
    $stmt->bind_param('sssss', $name, $email, $phone, $message, $ip);
    $stmt->execute();
    $stmt->close();
    $db->close();
}
