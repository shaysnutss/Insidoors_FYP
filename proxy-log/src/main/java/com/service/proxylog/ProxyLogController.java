package com.service.proxylog;

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

    @PutMapping("/suspectUpdate")
    public ResponseEntity<String> updateSuspectCase(@RequestBody String caseNo) throws JsonMappingException, JsonProcessingException {
        List<ProxyLog> logs = new ArrayList<>();
        ObjectMapper objectMapper = new ObjectMapper();
        //System.out.println("caseNo string: " + caseNo);

        String[] stringArray = caseNo.split("/");
        List<String> stringList = Arrays.asList(stringArray);

        for (int i = 0; i < stringList.size(); i++) {
          
            JsonNode jsonNode = objectMapper.readTree(stringList.get(i));
            //System.out.println("jsonnode: " + jsonNode);
            Optional<ProxyLog> optionalLog = proxyLogRepo.findById(jsonNode.get("id").asLong());
            ProxyLog existingLog = optionalLog.get();
            existingLog.setSuspect(jsonNode.get("suspect").asInt());
            logs.add(existingLog);
        }
        
        proxyLogRepo.saveAll(logs);
        return ResponseEntity.ok("Done");
    }
}
