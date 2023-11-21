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
import org.springframework.http.ResponseEntity;

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class TaskManagementServiceController {
    
    private final TaskManagementServiceRepository tMServiceRepo;

    @GetMapping("/tasks")
    public ResponseEntity<List<TaskManagementService>> getAllTasks() {
        List<TaskManagementService> tasks = tMServiceRepo.findAll();

        if (!tasks.isEmpty()) {
            return ResponseEntity.ok(tasks);
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @GetMapping("/tasks/{id}")
    public ResponseEntity<TaskManagementService> getTaskById(@PathVariable Long id) {
        Optional<TaskManagementService> taskOptional = tMServiceRepo.findById(id);

        if (taskOptional.isPresent()) {
            return ResponseEntity.ok(taskOptional.get());
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @GetMapping("/tasks/employee/{id}")
    public ResponseEntity<List<TaskManagementService>> getTaskByEmployeeId(@PathVariable int id) {
        List<TaskManagementService> tasks = tMServiceRepo.findAllByEmployeeId(id);

        if (!tasks.isEmpty()) {
            return ResponseEntity.ok(tasks);
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @GetMapping("/tasks/account/{id}")
    public ResponseEntity<List<TaskManagementService>> getTaskByAccountId(@PathVariable int id) {
        List<TaskManagementService> tasks = tMServiceRepo.findAllByAccountId(id);

        if (!tasks.isEmpty()) {
            return ResponseEntity.ok(tasks);
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/tasks")
    public List<TaskManagementService> addTask(@RequestBody String tm) throws JsonMappingException, JsonProcessingException{
        List<TaskManagementService> toSave = new ArrayList<>();
        ObjectMapper objectMapper = new ObjectMapper();
        
        String[] stringArray = tm.split("/");
        List<String> stringList = Arrays.asList(stringArray);

        for (int i = 0; i < stringList.size(); i++) {
            TaskManagementService tmService = new TaskManagementService();
            JsonNode jsonNode = objectMapper.readTree(stringList.get(i));

            tmService.setIncidentTitle(jsonNode.get("incidentTitle").asText());
            tmService.setIncidentDesc(jsonNode.get("incidentDesc").asText());
            //tmService.setIncidentTimestamp(jsonNode.get("incidentTimestamp").asText());
            tmService.setSeverity(jsonNode.get("severity").asInt());
            tmService.setStatus("Open");
            tmService.setAccountId(jsonNode.get("accountId").asInt());
            tmService.setEmployeeId(jsonNode.get("employeeId").asInt());
            tmService.setTruePositive(false);

            toSave.add(tmService);
            System.out.println(stringList.get(i));
        }
        

        return tMServiceRepo.saveAll(toSave);
    }

    @PutMapping("/statusUpdate/{id}")
    public ResponseEntity<TaskManagementService> updateTaskStatus(@PathVariable Long id, @RequestBody String tmNew) {
        Optional<TaskManagementService> optionalTask = tMServiceRepo.findById(id);

        if (optionalTask.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }

        TaskManagementService existingTask = optionalTask.get();

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(tmNew);

            existingTask.setId(id);
            existingTask.setStatus(jsonNode.get("status").asText());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body(null); 
        }

        TaskManagementService updatedTask = tMServiceRepo.save(existingTask);
        return ResponseEntity.ok(updatedTask);
    }

    @PutMapping("/accountUpdate/{id}")
    public ResponseEntity<TaskManagementService> updateTaskAccountId(@PathVariable Long id, @RequestBody String tmNew) {
        Optional<TaskManagementService> optionalTask = tMServiceRepo.findById(id);

        if (optionalTask.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }

        TaskManagementService existingTask = optionalTask.get();

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(tmNew);

            existingTask.setId(id);
            existingTask.setAccountId(jsonNode.get("accountId").asInt());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body(null); 
        }

        TaskManagementService updatedTask = tMServiceRepo.save(existingTask);
        return ResponseEntity.ok(updatedTask);
    }

    @PutMapping("/truePositiveUpdate/{id}")
    public ResponseEntity<TaskManagementService> updateTruePositive(@PathVariable Long id, @RequestBody String tmNew) {
        Optional<TaskManagementService> optionalTask = tMServiceRepo.findById(id);

        if (optionalTask.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }

        TaskManagementService existingTask = optionalTask.get();

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(tmNew);

            existingTask.setId(id);
            existingTask.setTruePositive(jsonNode.get("truePositive").asBoolean());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body(null); 
        }

        TaskManagementService updatedTask = tMServiceRepo.save(existingTask);
        return ResponseEntity.ok(updatedTask);
    }

    @PutMapping("/descUpdate/{id}")
    public ResponseEntity<TaskManagementService> updateTaskDesc(@PathVariable Long id, @RequestBody String tmNew) {
        Optional<TaskManagementService> optionalTask = tMServiceRepo.findById(id);

        if (optionalTask.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }

        TaskManagementService existingTask = optionalTask.get();

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(tmNew);

            existingTask.setId(id);
            existingTask.setIncidentDesc(jsonNode.get("incidentDesc").asText());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body(null); 
        }

        TaskManagementService updatedTask = tMServiceRepo.save(existingTask);
        return ResponseEntity.ok(updatedTask);
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
