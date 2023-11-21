package com.service.proxylog;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class ProxyLogController {
    
    private final ProxyLogRepository proxyLogRepo;

    @GetMapping("/proxylogs")
    public ResponseEntity<List<ProxyLog>> getAllProxyLogs() {
        List<ProxyLog> logs = proxyLogRepo.findAll();

        if (logs.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(logs);
        }
    }

    @GetMapping("/proxylogs/{id}")
    public ResponseEntity<List<ProxyLog>> getAllProxyLogsAfterId(@PathVariable Long id) {
        List<ProxyLog> logs = proxyLogRepo.findAllAfterId(id);

        if (logs.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(logs);
        }
    }
}
