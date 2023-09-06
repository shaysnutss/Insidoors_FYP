package com.service.behavioralanalysisservice;

import java.util.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.HttpStatus;

@RestController
public class BehavioralAnalysisServiceController {
    
    @Autowired
    private BehavioralAnalysisServiceRepository bAServiceRepo;

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

    @PutMapping("/riskrating/{id}")
    public BehavioralAnalysisService updateRiskRatingByEmployeeId(@PathVariable int id, @RequestBody BehavioralAnalysisService baNew){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);
        
        if (ba == null) {
            throw new BehavioralAnalysisNotFoundException(id);
        }
        
        ba.setId(ba.getId());
        ba.setEmployeeId(id);
        ba.setRiskRating(baNew.getRiskRating());
        ba.setSuspectedCases(ba.getSuspectedCases());
        return bAServiceRepo.save(ba);
    }

    @PutMapping("/suspectedcases/{id}")
    public BehavioralAnalysisService updateSuspectedCasesByEmployeeId(@PathVariable int id, @RequestBody BehavioralAnalysisService baNew){
        BehavioralAnalysisService ba = bAServiceRepo.findByEmployeeId(id);
        
        if (ba == null) {
            throw new BehavioralAnalysisNotFoundException(id);
        }
        
        ba.setId(ba.getId());
        ba.setEmployeeId(id);
        ba.setRiskRating(ba.getRiskRating());
        ba.setSuspectedCases(baNew.getSuspectedCases());
        return bAServiceRepo.save(ba);
    }

    @DeleteMapping("/behavioralanalysis/{id}")
    public void deleteBehavioralAnalysis(@PathVariable Long id){
        try {
            bAServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            // error message
        }
    }

}
