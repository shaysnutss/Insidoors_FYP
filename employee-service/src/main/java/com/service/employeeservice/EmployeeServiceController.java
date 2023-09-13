package com.service.employeeservice;

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
public class EmployeeServiceController {
    
    @Autowired
    private EmployeeServiceRepository employeeServiceRepo;

    @GetMapping("/employees")
    public List<EmployeeService> getAllEmployees(){
        return employeeServiceRepo.findAll();
    }

    @GetMapping("/employees/{id}")
    public Optional<EmployeeService> getEmployeeById(@PathVariable Long id){
        return employeeServiceRepo.findById(id);
    }
}
