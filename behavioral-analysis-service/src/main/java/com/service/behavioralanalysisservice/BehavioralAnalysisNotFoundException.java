package com.service.behavioralanalysisservice;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND) 
public class BehavioralAnalysisNotFoundException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;

    public BehavioralAnalysisNotFoundException(int id) {
        super("Could not find the behavioral analysis with employee id: " + id);
    }
}
