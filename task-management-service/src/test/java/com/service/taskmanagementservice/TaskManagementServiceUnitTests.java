package com.service.taskmanagementservice;

import static org.mockito.Mockito.*;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;

import java.time.*;
import java.util.*;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(TaskManagementServiceController.class)
public class TaskManagementServiceUnitTests {
    
    @MockBean private TaskManagementServiceRepository tMServiceRepo;
    @Autowired private MockMvc mockMvc;
    @Autowired private ObjectMapper objectMapper;

    @Test 
    void shouldCreateTask() throws Exception {
        TaskManagementService task = new TaskManagementService(1L, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false);

        mockMvc.perform(post("/api/v1/tasks").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(task)))
            .andExpect(status().isCreated())
            .andDo(print());
    }

    @Test
    void shouldReturnListOfTasks() throws Exception {
        List<TaskManagementService> mockTasks = new ArrayList<>(
            Arrays.asList(new TaskManagementService(1L, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false),
            new TaskManagementService(2L, "B23", "second", "description", "21-12-2023", 12, 120, "Closed", 34, false)));
    
        when(tMServiceRepo.findAll()).thenReturn(mockTasks);
        mockMvc.perform(get("/api/v1/tasks"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockTasks.size()))
            .andDo(print());
    }

    @Test
    void shouldReturnTask() throws Exception {
        long id = 1L;
        TaskManagementService mockTask = new TaskManagementService(id, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false);

        when(tMServiceRepo.findById(any())).thenReturn(Optional.of(mockTask));
        mockMvc.perform(get("/api/v1/tasks/1"))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.incidentTitle").value(mockTask.getIncidentTitle()))
            .andExpect(jsonPath("$.incidentTimestamp").value(mockTask.getIncidentTimestamp().toString()))
            .andExpect(jsonPath("$.employeeId").value(mockTask.getEmployeeId()))
            .andExpect(jsonPath("$.severity").value(mockTask.getSeverity()))
            .andExpect(jsonPath("$.status").value(mockTask.getStatus()))
            .andExpect(jsonPath("$.accountId").value(mockTask.getAccountId()))
            .andDo(print());
    }

    @Test
    void shouldReturnTaskByEmployeeId() throws Exception {
        long id = 1L;
        List<TaskManagementService> mockTasks = new ArrayList<>(
            Arrays.asList(new TaskManagementService(id, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false)));

        when(tMServiceRepo.findAllByEmployeeId(anyInt())).thenReturn(mockTasks);
        mockMvc.perform(get("/api/v1/tasks/employee/23"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockTasks.size()))
            .andDo(print());
    }

    @Test
    void shouldReturnTaskByAccountId() throws Exception {
        long id = 1L;
        List<TaskManagementService> mockTasks = new ArrayList<>(
            Arrays.asList(new TaskManagementService(id, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false)));

        when(tMServiceRepo.findAllByAccountId(anyInt())).thenReturn(mockTasks);
        mockMvc.perform(get("/api/v1/tasks/account/12"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockTasks.size()))
            .andDo(print());
    }

    @Test
    void shouldUpdateTaskStatus() throws Exception {
        long id = 1L;

        TaskManagementService mockTask = new TaskManagementService(id, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false);
        TaskManagementService mockTaskUpdate = new TaskManagementService(id, "B23", "new", "description", "21-12-2023", 23, 200, "Closed", 12, false);

        when(tMServiceRepo.findById(id)).thenReturn(Optional.of(mockTask));
        when(tMServiceRepo.save(any(TaskManagementService.class))).thenReturn(mockTaskUpdate);

        mockMvc.perform(put("/api/v1/statusUpdate/1").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockTaskUpdate)))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.incidentTitle").value(mockTaskUpdate.getIncidentTitle()))
            .andExpect(jsonPath("$.incidentTimestamp").value(mockTaskUpdate.getIncidentTimestamp().toString()))
            .andExpect(jsonPath("$.employeeId").value(mockTaskUpdate.getEmployeeId()))
            .andExpect(jsonPath("$.severity").value(mockTaskUpdate.getSeverity()))
            .andExpect(jsonPath("$.status").value(mockTaskUpdate.getStatus()))
            .andExpect(jsonPath("$.accountId").value(mockTaskUpdate.getAccountId()))
            .andDo(print());
    }

    @Test
    void shouldUpdateTaskSOC() throws Exception {
        long id = 1L;

        TaskManagementService mockTask = new TaskManagementService(id, "B23", "first", "description", "21-12-2023", 23, 200, "Open", 12, false);
        TaskManagementService mockTaskUpdate = new TaskManagementService(id, "B23", "new", "description", "21-12-2023", 23, 200, "Open", 8, false);

        when(tMServiceRepo.findById(id)).thenReturn(Optional.of(mockTask));
        when(tMServiceRepo.save(any(TaskManagementService.class))).thenReturn(mockTaskUpdate);

        mockMvc.perform(put("/api/v1/accountUpdate/1").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockTaskUpdate)))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.incidentTitle").value(mockTaskUpdate.getIncidentTitle()))
            .andExpect(jsonPath("$.incidentTimestamp").value(mockTaskUpdate.getIncidentTimestamp().toString()))
            .andExpect(jsonPath("$.employeeId").value(mockTaskUpdate.getEmployeeId()))
            .andExpect(jsonPath("$.severity").value(mockTaskUpdate.getSeverity()))
            .andExpect(jsonPath("$.status").value(mockTaskUpdate.getStatus()))
            .andExpect(jsonPath("$.accountId").value(mockTaskUpdate.getAccountId()))
            .andDo(print());
    }

    @Test
    void shouldDeleteTask() throws Exception {
        long id = 1L;

        doNothing().when(tMServiceRepo).deleteById(id);
        mockMvc.perform(delete("/api/v1/tasks/1"))
            .andExpect(status().isNoContent())
            .andDo(print());
    }
}
