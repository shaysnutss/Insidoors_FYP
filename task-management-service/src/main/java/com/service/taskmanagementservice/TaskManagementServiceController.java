package com.service.taskmanagementservice;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class TaskManagementServiceController {
    
    private final TaskManagementServiceRepository tMServiceRepo;

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
        return tMServiceRepo.findAllByEmployeeId(id);
    }

    @GetMapping("/tasks/account/{id}")
    public List<TaskManagementService> getTaskByAccountId(@PathVariable int id){
        return tMServiceRepo.findAllByAccountId(id);
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/tasks")
    public TaskManagementService addTask(@RequestBody TaskManagementService tm){
        return tMServiceRepo.save(tm);
    }

    @PutMapping("/statusUpdate/{id}")
    public TaskManagementService updateTaskStatus(@PathVariable Long id, @RequestBody String tmNew){
        Optional<TaskManagementService> tm = tMServiceRepo.findById(id);
        try {       
            if (tm == null) {
                throw new TaskManagementNotFoundException(id);
            }

            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(tmNew);

            tm.get().setId(id);
            tm.get().setStatus(jsonNode.get("status").asText());
        
        } catch (JsonMappingException e) {
            
            e.printStackTrace();
        } catch (JsonProcessingException e) {
            
            e.printStackTrace();
        }
        
        return tMServiceRepo.save(tm.get());
    }

    @PutMapping("/accountUpdate/{id}")
    public TaskManagementService updateTaskAccountId(@PathVariable Long id, @RequestBody String tmNew){
        Optional<TaskManagementService> tm = tMServiceRepo.findById(id);
        try {       
            if (tm == null) {
                throw new TaskManagementNotFoundException(id);
            }

            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(tmNew);

            tm.get().setId(id);
            tm.get().setAccountId(jsonNode.get("accountId").asInt());
        
        } catch (JsonMappingException e) {
            
            e.printStackTrace();
        } catch (JsonProcessingException e) {
            
            e.printStackTrace();
        }
        
        return tMServiceRepo.save(tm.get());
    }

    @DeleteMapping("/tasks/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteTask(@PathVariable Long id){
        try {
            tMServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            throw new TaskManagementNotFoundException(id);
        }
    }
}
