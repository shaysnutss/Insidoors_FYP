package com.service.behavioralanalysiscompositeservice.Controller;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.service.behavioralanalysiscompositeservice.Service.BehavioralAnalysisCompositeService;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1/BAComposite")
public class BehavioralAnalysisCompositeController {

    @Value("${EMPLOYEES_API_BASE_URL}")
    private String employeesApiBaseUrl;

    @Value("${BHATOMIC_API_BASE_URL}")
    private String bhApiBaseUrl;

    public final BehavioralAnalysisCompositeService behavioralAnalysisCompositeService;

    /*
    *1. Showing all employees
    1. Api request to behave comp (actually account but I so this later)
    2. Get all employees api request to employee service
    3. Get all behav analysis to behave analysis atomic
    4. Put the severity rating of each employee tgt w rest of each employee info
    5. Send response back to frontend
    * */

    @GetMapping("/viewAllEmployees")
    public ResponseEntity<?> viewAllEmployees(){

        // Using getMethod service method to send GET request to employee service to get all employees
        List<Map<String, Object>> employeesResponse = behavioralAnalysisCompositeService.getMethod(employeesApiBaseUrl);
        if (employeesResponse == null){
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(null);
        }
        // return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        // Using getMethod service method to send GET request to bh atomic service to get all bh
        List<Map<String, Object>> bhResponse = behavioralAnalysisCompositeService.getMethod(bhApiBaseUrl);

        List<Map<String, Object>> toReturn = new ArrayList<>();

        for (Map<String, Object> employee : employeesResponse) {
            int employeeId = (int) employee.get("id");
            Map<String,Object> relevantBH = behavioralAnalysisCompositeService.employeeBH(employeeId,bhResponse);

            // if all employees have not committed any incidents
            // or if just the relevant employee has not committed any incidents
            if((bhResponse == null) || (relevantBH == null)){
                employee.put("riskRating",0);
                employee.put("suspectedCases",0);
            }else{
                employee.put("riskRating",relevantBH.get("riskRating"));
                employee.put("suspectedCases",relevantBH.get("suspectedCases"));
            }
            toReturn.add(employee);

        }
        return ResponseEntity.ok(toReturn);

    }






















}
