package com.service.behavioralanalysisservice;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.*;

import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class BehavioralAnalysisServiceController {
    
    private final BehavioralAnalysisServiceRepository bAServiceRepo;

    @GetMapping("/behavioralanalysis")
    public ResponseEntity<List<BehavioralAnalysisService>> getAllBehavioralAnalysis() {
        List<BehavioralAnalysisService> analysisList = bAServiceRepo.findAll();

        if (analysisList.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(analysisList);
        }
    }

    @GetMapping("/behavioralanalysis/{id}")
    public ResponseEntity<BehavioralAnalysisService> getBehavioralAnalysisById(@PathVariable Long id) {
        Optional<BehavioralAnalysisService> analysisOptional = bAServiceRepo.findById(id);

        if (analysisOptional.isPresent()) {
            return ResponseEntity.ok(analysisOptional.get());
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @GetMapping("/riskrating/{id}")
    public ResponseEntity<Integer> getRiskRatingByEmployeeId(@PathVariable int id) {
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);

        if (ba != null) {
            return ResponseEntity.ok(ba.getRiskRating());
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @GetMapping("/suspectedcases/{id}")
    public ResponseEntity<Integer> getSuspectedCasesByEmployeeId(@PathVariable int id){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);
        
        if (ba != null) {
            return ResponseEntity.ok(ba.getSuspectedCases());
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/behavioralanalysis")
    public BehavioralAnalysisService addBehavioralAnalysis(@RequestBody BehavioralAnalysisService ba){
        return bAServiceRepo.save(ba);
    }

    @PutMapping("/updateRiskRating/{id}")
    public ResponseEntity<BehavioralAnalysisService> updateRiskRatingByEmployeeId(@PathVariable int id, @RequestBody String baNew){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);

        try {       
            if (ba == null) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
            }

            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(baNew);

            ba.setId(ba.getId());
            ba.setEmployeeId(id);
            ba.setRiskRating(ba.getRiskRating() + jsonNode.get("riskRating").asInt());
        
        } catch (JsonMappingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();

        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        }
        
        bAServiceRepo.save(ba);
        return ResponseEntity.ok(ba);
    }

    @PutMapping("/updateSuspectedCases/{id}")
    public ResponseEntity<BehavioralAnalysisService> updateSuspectedCasesByEmployeeId(@PathVariable int id, @RequestBody String baNew){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);

        try {       
            if (ba == null) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
            }

            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(baNew);

            ba.setId(ba.getId());
            ba.setEmployeeId(id);
            ba.setSuspectedCases(jsonNode.get("suspectedCases").asInt());
        
        } catch (JsonMappingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        }
        
        bAServiceRepo.save(ba);
        return ResponseEntity.ok(ba);
    }

    @DeleteMapping("/behavioralanalysis/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteBehavioralAnalysis(@PathVariable Long id){
        try {
            bAServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            throw new BehavioralAnalysisNotFoundException(id);
        }
    }

}