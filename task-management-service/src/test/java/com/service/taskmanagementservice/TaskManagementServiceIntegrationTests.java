package com.service.taskmanagementservice;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.*;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.jdbc.Sql;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(locations = "classpath:application-test.properties")
public class TaskManagementServiceIntegrationTests {
    
    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    @Autowired
    private TaskManagementServiceRepository tMServiceRepo;

    private static HttpHeaders headers;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @BeforeAll
    public static void init() {
        headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
    }

    private String createURLWithPort() {
        return "http://localhost:" + port + "/api/v1";
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testTaskList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<TaskManagementService>>(){});
        List<TaskManagementService> taskList = response.getBody();
        assertNotNull(taskList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(taskList.size(), tMServiceRepo.findAll().size());
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testTaskById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<TaskManagementService>>(){});
        Optional<TaskManagementService> task = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(task, tMServiceRepo.findById(1L));
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testTaskListByEmployeeId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/employee/23"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<TaskManagementService>>(){});
        List<TaskManagementService> taskList = response.getBody();
        assertNotNull(taskList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(taskList.size(), tMServiceRepo.findAll().size());
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testTaskListByAccountId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/account/12"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<TaskManagementService>>(){});
        List<TaskManagementService> taskList = response.getBody();
        assertNotNull(taskList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(taskList.size(), tMServiceRepo.findAll().size());
    }

    @Test
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCreateTask() throws JsonProcessingException {
        TaskManagementService task = new TaskManagementService(1L, "first", null, 23, 200, "Open", 12, null);

        HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(task), headers);
        ResponseEntity<TaskManagementService> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks"), HttpMethod.POST, entity, TaskManagementService.class);
            
        assertEquals(response.getStatusCode(), HttpStatus.CREATED);
        TaskManagementService taskRes = Objects.requireNonNull(response.getBody());
        assertEquals(taskRes.getStatus(), tMServiceRepo.save(task).getStatus());
        assertEquals(taskRes.getIncidentTitle(), tMServiceRepo.save(task).getIncidentTitle());
        assertEquals(taskRes.getSeverity(), tMServiceRepo.save(task).getSeverity());
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testUpdateStatus() throws JsonProcessingException {
        TaskManagementService task = new TaskManagementService();
        task.setStatus("Closed");
        task.setId(1L);

        ObjectMapper mapper = new ObjectMapper();
        String requestBody = mapper.writeValueAsString(task);

        HttpEntity<String> entity = new HttpEntity<>(requestBody, null);

        ResponseEntity<TaskManagementService> response = restTemplate.exchange(
            (createURLWithPort() + "/statusUpdate/1"), HttpMethod.PUT, entity, TaskManagementService.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals("Closed", response.getBody().getStatus());
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM task_management_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testUpdateAccountId() throws JsonProcessingException {
        TaskManagementService task = new TaskManagementService();
        task.setAccountId(9);
        task.setId(1L);

        ObjectMapper mapper = new ObjectMapper();
        String requestBody = mapper.writeValueAsString(task);

        HttpEntity<String> entity = new HttpEntity<>(requestBody, null);

        ResponseEntity<TaskManagementService> response = restTemplate.exchange(
            (createURLWithPort() + "/accountUpdate/1"), HttpMethod.PUT, entity, TaskManagementService.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals(9, response.getBody().getAccountId());
    }

    @Test
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first comment\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    public void testDeleteTask() {
        ResponseEntity<Void> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/1"), HttpMethod.DELETE, null, Void.class);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }
}
