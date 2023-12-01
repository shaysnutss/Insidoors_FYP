package com.service.pcaccess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
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

    @PutMapping("/suspectUpdate")
    public ResponseEntity<String> updateSuspectCase(@RequestBody String caseNo) throws JsonMappingException, JsonProcessingException {
        List<PcAccess> logs = new ArrayList<>();
        ObjectMapper objectMapper = new ObjectMapper();

        String[] stringArray = caseNo.split("/");
        List<String> stringList = Arrays.asList(stringArray);

        for (int i = 0; i < stringList.size(); i++) {
          
            JsonNode jsonNode = objectMapper.readTree(stringList.get(i));
            Optional<PcAccess> optionalLog = pcAccessRepo.findById(jsonNode.get("id").asLong());
            PcAccess existingLog = optionalLog.get();
            existingLog.setSuspect(jsonNode.get("suspect").asInt());
            logs.add(existingLog);
        }
        
        pcAccessRepo.saveAll(logs);
        return ResponseEntity.ok("Done");
    }
    
}
