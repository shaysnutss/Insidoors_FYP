package com.service.pcaccess;

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
public class PcAccessController {

    private final PcAccessRepository pcAccessRepo;

    @GetMapping("/pcaccesslogs")
    public ResponseEntity<List<PcAccess>> getAllPcAccessLogs() {
        List<PcAccess> logs = pcAccessRepo.findAll();

        if (logs.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(logs);
        }
    }

    @GetMapping("/pcaccesslogs/{id}")
    public ResponseEntity<List<PcAccess>> getAllPcAccessLogsAfterId(@PathVariable Long id) {
        List<PcAccess> logs = pcAccessRepo.findAllAfterId(id);

        if (logs.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(logs);
        }
    }
    
}
