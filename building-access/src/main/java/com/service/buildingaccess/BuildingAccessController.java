package com.service.buildingaccess;

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
public class BuildingAccessController {

    private final BuildingAccessRepository buildingAccessRepo;

    @GetMapping("/buildingaccesslogs")
    public ResponseEntity<List<BuildingAccess>> getAllBuildingAccessLogs() {
        List<BuildingAccess> logs = buildingAccessRepo.findAll();

        if (logs.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(logs);
        }
    }

    @GetMapping("/buildingaccesslogs/{id}")
    public ResponseEntity<List<BuildingAccess>> getAllBuildingAccessLogsAfterId(@PathVariable Long id) {
        List<BuildingAccess> logs = buildingAccessRepo.findAllAfterId(id);

        if (logs.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(logs);
        }
    }
    
}
