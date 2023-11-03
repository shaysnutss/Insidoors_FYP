package com.service.accountservice.Controller;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.service.accountservice.Service.accountService;

import org.apache.catalina.connector.Response;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;

import lombok.RequiredArgsConstructor;

@RestController
@CrossOrigin(origins = "http://localhost:30008")
@RequestMapping(path = "/api/v1/accountByBA")
public class BehavioralAnalysisComposite {

    @Value("${BA_COMPOSITE_API_BASE_URL}")
    private String baCompositeApiBaseURL; 

    private accountService accountService; 

    public BehavioralAnalysisComposite(com.service.accountservice.Service.accountService accountService){
        this.accountService = accountService; 
    }

    /**
     * Find all employees with their behaviour analysis
     *
     * @return list of all employees with their behaviour analysis 
     */
    @GetMapping("/viewAllEmployees")
    public ResponseEntity<?> viewAllEmployees(){
        return accountService.getMethod(baCompositeApiBaseURL + "/viewAllEmployees");
    }


    /**
     * Find the incidents of an employee 
     * @return list the incidents of an employee
     */

    @GetMapping("/viewIncidentsByEmployeeId/{id}")
    public ResponseEntity<?> viewIncidentsByEmployeeId(@PathVariable(value =  "id") int id){
        return accountService.getMethod(baCompositeApiBaseURL + "/viewIncidentsByEmployeeId/" + id);
    }

    @GetMapping("/viewEmployeeByName/{name}")
    public ResponseEntity<?> viewEmployeeById(@PathVariable(value = "name" ) String name){
        return accountService.getMethod(baCompositeApiBaseURL + "/viewEmployeeByName/" + name);
    }




    
}
