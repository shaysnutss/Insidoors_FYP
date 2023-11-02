package com.service.employeeservice;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface EmployeeServiceRepository extends JpaRepository<EmployeeService, Long> {
    
    EmployeeService findByfirstname(String name);
    
}
