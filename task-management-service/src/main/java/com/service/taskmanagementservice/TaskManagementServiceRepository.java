package com.service.taskmanagementservice;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TaskManagementServiceRepository extends JpaRepository<TaskManagementService, Long> {
    
    List<TaskManagementService> findAllByEmployeeId(int employeeId);

    List<TaskManagementService> findAllByAccountId(int accountId);
}
