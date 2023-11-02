package com.service.behavioralanalysiscompositeservice.Controller;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.service.behavioralanalysiscompositeservice.Service.BehavioralAnalysisCompositeService;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;

@RestController
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:30008")
@RequestMapping(path = "/api/v1/BAComposite")
public class BehavioralAnalysisCompositeController {

    @Value("${EMPLOYEES_API_BASE_URL}")
    private String employeesApiBaseUrl;

    @Value("${BHATOMIC_API_BASE_URL}")
    private String bhApiBaseUrl;

    @Value("${TASK_COMP_API_BASE_URL}")
    private String taskMCompApiBaseUrl;

    public final BehavioralAnalysisCompositeService behavioralAnalysisCompositeService;

    /**
     * Find all employees with their behaviour analysis
     *
     * @return list of all employees with their behaviour analysis
     */

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

    /**
     * Find an employee with their behaviour analysis
     *
     * @return an employee with their behaviour analysis + list of incidents of each
     *         employee
     */

    @GetMapping("/viewIncidentsByEmployeeId/{id}")
    public ResponseEntity<?> viewAllEmployees(@PathVariable(value = "id") long id){

        //Get all tasks by employee ic
        List<Map<String, Object>> incidentsByEmployee = behavioralAnalysisCompositeService.getMethod( taskMCompApiBaseUrl + "/viewAllTasksByEmployeeId/" + id);
        return ResponseEntity.ok(incidentsByEmployee);


    }

    /**
     * Find an employee with their behaviour analysis by name
     *
     * @return an employee with their behaviour analysis
     */
    @GetMapping("/viewEmployeeByName/{name}")
    public ResponseEntity<?> viewEmployeeById(@PathVariable(value = "name") String name) {
        // get the id by name
        Map<String,Object> employeeIdObject = behavioralAnalysisCompositeService.getEmployeeMethod(employeesApiBaseUrl + "/getEmployee/" + name);
        String id = "" + employeeIdObject.get("id"); 
        Map<String, Object> employee = behavioralAnalysisCompositeService.getEmployeeMethod(employeesApiBaseUrl + "/" + id);
        if (employee == null){
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
        
        return ResponseEntity.ok(employee); 
    }























}
