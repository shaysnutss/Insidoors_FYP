package com.service.employeeservice;

import java.util.*;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class EmployeeServiceController {
    
    private final EmployeeServiceRepository employeeServiceRepo;

    @GetMapping("/employees")
    public List<EmployeeService> getAllEmployees(){
        return employeeServiceRepo.findAll();
    }

    @GetMapping("/employees/{id}")
    public Optional<EmployeeService> getEmployeeById(@PathVariable Long id){
        return employeeServiceRepo.findById(id);
    }
}
