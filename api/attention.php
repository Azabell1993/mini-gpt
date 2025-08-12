<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit(0);
}

function softmax($x) {
    $max = max($x);
    $exp_values = array_map(function($val) use ($max) {
        return exp($val - $max);
    }, $x);
    $sum = array_sum($exp_values);
    return array_map(function($val) use ($sum) {
        return $val / $sum;
    }, $exp_values);
}

function attention_demo($q, $k, $v, $causal = true) {
    $L = count($q);
    $dk = count($q[0]);
    $dv = count($v[0]);
    $scale = 1.0 / sqrt($dk);
    
    // 점수 계산
    $scores = [];
    for ($t = 0; $t < $L; $t++) {
        $scores[$t] = [];
        for ($u = 0; $u < $L; $u++) {
            $dot = 0.0;
            for ($i = 0; $i < $dk; $i++) {
                $dot += $q[$t][$i] * $k[$u][$i];
            }
            $scores[$t][$u] = $dot * $scale;
            if ($causal && $u > $t) {
                $scores[$t][$u] = -1e9; // 인과적 마스크
            }
        }
    }
    
    // 소프트맥스 적용
    $attn = [];
    for ($t = 0; $t < $L; $t++) {
        $attn[$t] = softmax($scores[$t]);
    }
    
    // 출력 계산
    $out = [];
    for ($t = 0; $t < $L; $t++) {
        $out[$t] = [];
        for ($j = 0; $j < $dv; $j++) {
            $s = 0.0;
            for ($u = 0; $u < $L; $u++) {
                $s += $attn[$t][$u] * $v[$u][$j];
            }
            $out[$t][$j] = $s;
        }
    }
    
    return [
        'attention_weights' => $attn,
        'output' => $out,
        'scores' => $scores
    ];
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $input = json_decode(file_get_contents('php://input'), true);
    
    if (!isset($input['q']) || !isset($input['k']) || !isset($input['v'])) {
        http_response_code(400);
        echo json_encode(['error' => 'Missing q, k, or v matrices']);
        exit;
    }
    
    $q = $input['q'];
    $k = $input['k'];
    $v = $input['v'];
    $causal = isset($input['causal']) ? $input['causal'] : true;
    
    $result = attention_demo($q, $k, $v, $causal);
    echo json_encode($result);
    
} else {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
}
?>
