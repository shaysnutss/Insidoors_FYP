package com.service.behavioralanalysisservice;

import java.util.*;

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

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class BehavioralAnalysisServiceController {
    
    private final BehavioralAnalysisServiceRepository bAServiceRepo;

    @GetMapping("/behavioralanalysis")
    public List<BehavioralAnalysisService> getAllBehavioralAnalysis(){
        return bAServiceRepo.findAll();
    }

    @GetMapping("/behavioralanalysis/{id}")
    public Optional<BehavioralAnalysisService> getBehavioralAnalysisById(@PathVariable Long id){
        return bAServiceRepo.findById(id);
    }

    @GetMapping("/riskrating/{id}")
    public int getRiskRatingByEmployeeId(@PathVariable int id){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);
        
        if (ba == null) {
            throw new BehavioralAnalysisNotFoundException(id);
        }
        
        return ba.getRiskRating();
    }

    @GetMapping("/suspectedcases/{id}")
    public int getSuspectedCasesByEmployeeId(@PathVariable int id){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);
        
        if (ba == null) {
            throw new BehavioralAnalysisNotFoundException(id);
        }
        
        return ba.getSuspectedCases();
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/behavioralanalysis")
    public BehavioralAnalysisService addBehavioralAnalysis(@RequestBody BehavioralAnalysisService ba){
        return bAServiceRepo.save(ba);
    }

    @PutMapping("/behavioralanalysis/{id}")
    public BehavioralAnalysisService updateByEmployeeId(@PathVariable int id, @RequestBody BehavioralAnalysisService baNew){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);
        
        if (ba == null) {
            throw new BehavioralAnalysisNotFoundException(id);
        }
        
        ba.setId(ba.getId());
        ba.setEmployeeId(id);
        ba.setRiskRating(baNew.getRiskRating());
        ba.setSuspectedCases(baNew.getSuspectedCases());
        return bAServiceRepo.save(ba);
    }

    @DeleteMapping("/behavioralanalysis/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteBehavioralAnalysis(@PathVariable Long id){
        try {
            bAServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            // error message
        }
    }

}
