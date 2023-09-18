package com.service.employeeservice;

import static org.mockito.Mockito.*;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;

import java.time.LocalDate;
import java.time.Month;
import java.util.*;


import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(EmployeeServiceController.class)
public class EmployeeServiceTests {

    @MockBean private EmployeeServiceRepository employeeServiceRepo;
    @Autowired private MockMvc mockMvc;

    @Test
    void shouldReturnListOfEmployees() throws Exception {
        List<EmployeeService> mockEmployees = new ArrayList<>(
            Arrays.asList(new EmployeeService(1L, "Mary", "Elizabeth", "marye@gmail.com", "female", "Asset Management", LocalDate.of(2017,Month.FEBRUARY,3), null, "Seattle"),
            new EmployeeService(2L, "Sean", "Henry", "seanh@gmail.com", "male", "Asset Management", LocalDate.of(2017,Month.FEBRUARY,3), null, "London")));
    
        when(employeeServiceRepo.findAll()).thenReturn(mockEmployees);
        mockMvc.perform(get("/api/v1/employees"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockEmployees.size()))
            .andDo(print());
    }

    @Test
    void shouldReturnEmployee() throws Exception {
        long id = 1L;
        EmployeeService mockEmployee = new EmployeeService(id, "Mary", "Elizabeth", "marye@gmail.com", "female", "Asset Management", LocalDate.of(2017,Month.FEBRUARY,3), null, "Seattle");

        when(employeeServiceRepo.findById(any())).thenReturn(Optional.of(mockEmployee));
        mockMvc.perform(get("/api/v1/employees/1"))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.firstname").value(mockEmployee.getFirstname()))
            .andExpect(jsonPath("$.lastname").value(mockEmployee.getLastname()))
            .andExpect(jsonPath("$.email").value(mockEmployee.getEmail()))
            .andExpect(jsonPath("$.gender").value(mockEmployee.getGender()))
            .andExpect(jsonPath("$.businessUnit").value(mockEmployee.getBusinessUnit()))
            .andExpect(jsonPath("$.location").value(mockEmployee.getLocation()))
            .andExpect(jsonPath("$.joinedDate").value(mockEmployee.getJoinedDate().toString()))
            .andDo(print());
    }
}
