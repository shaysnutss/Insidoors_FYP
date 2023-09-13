package com.service.taskmanagementservice;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND) 
public class TaskManagementNotFoundException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;

    public TaskManagementNotFoundException(Long id) {
        super("Could not find the task with id: " + id);
    }
}
