package com.service.employeeservice;

import java.util.*;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
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
    public ResponseEntity<List<EmployeeService>> getAllEmployees() {
        List<EmployeeService> employees = employeeServiceRepo.findAll();

        if (!employees.isEmpty()) {
            return ResponseEntity.ok(employees);
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @GetMapping("/employees/{id}")
    public ResponseEntity<EmployeeService> getEmployeeById(@PathVariable Long id) {
        Optional<EmployeeService> employeeOptional = employeeServiceRepo.findById(id);

        if (employeeOptional.isPresent()) {
            return ResponseEntity.ok(employeeOptional.get());
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }
}
