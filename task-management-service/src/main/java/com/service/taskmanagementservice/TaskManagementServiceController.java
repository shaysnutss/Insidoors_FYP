package com.service.taskmanagementservice;

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
public class TaskManagementServiceController {
    
    @Autowired
    private TaskManagementServiceRepository tMServiceRepo;

    @GetMapping("/tasks")
    public List<TaskManagementService> getAllTasks(){
        return tMServiceRepo.findAll();
    }

    @GetMapping("/tasks/{id}")
    public Optional<TaskManagementService> getTaskById(@PathVariable Long id){
        return tMServiceRepo.findById(id);
    }

    @GetMapping("/tasks/employee/{id}")
    public List<TaskManagementService> getTaskByEmployeeId(@PathVariable int id){
        return tMServiceRepo.findByEmployeeId(id);
    }

    @GetMapping("/tasks/account/{id}")
    public List<TaskManagementService> getTaskByAccountId(@PathVariable int id){
        return tMServiceRepo.findByAccountId(id);
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/tasks")
    public TaskManagementService addTask(@RequestBody TaskManagementService tm){
        return tMServiceRepo.save(tm);
    }

    @PutMapping("/tasks/{id}")
    public TaskManagementService updateTaskStatus(@PathVariable Long id, @RequestBody TaskManagementService tmNew){
        Optional<TaskManagementService> tm = tMServiceRepo.findById(id);
        
        if (tm == null) {
            throw new TaskManagementNotFoundException(id);
        }
        
        tm.get().setId(id);
        tm.get().setStatus(tmNew.getStatus());
        return tMServiceRepo.save(tm.get());
    }

    @DeleteMapping("/tasks/{id}")
    public void deleteTask(@PathVariable Long id){
        try {
            tMServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            throw new TaskManagementNotFoundException(id);
        }
    }
}