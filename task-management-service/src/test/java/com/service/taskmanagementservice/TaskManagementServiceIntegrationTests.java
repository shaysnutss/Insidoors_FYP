package com.service.taskmanagementservice;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.zaxxer.hikari.HikariDataSource;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.*;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.jdbc.Sql;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import jakarta.activation.DataSource;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(locations = "classpath:application-test.properties")
@Testcontainers
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TaskManagementServiceIntegrationTests {

    private static DataSource dataSource;
    
    @LocalServerPort
    private int port;

    @Container
    public static MySQLContainer<?> mySQLContainer = new MySQLContainer<>(DockerImageName.parse("mysql:8.0.26"))
        .withDatabaseName("behavioral_analysis")
        .withUsername("test")
        .withPassword("test")
        .waitingFor(Wait.forListeningPort())
        .withEnv("MYSQL_ROOT_HOST", "%");

    @DynamicPropertySource
    static void registerPgProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", mySQLContainer::getJdbcUrl);
        registry.add("spring.datasource.password", mySQLContainer::getPassword);
        registry.add("spring.datasource.username", mySQLContainer::getUsername);
    }

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

    @AfterAll
    static void tearDown() {
        if (dataSource instanceof HikariDataSource) {
            ((HikariDataSource) dataSource).close();
        }
    }

    private String createURLWithPort() {
        return "http://localhost:" + port + "/api/v1";
    }

    @Test
    @Order(1)
    @Sql(statements = "INSERT INTO task_management_service (tm_id, incident_title, incident_timestamp, employee_id, severity, status, account_id, date_assigned) values (1, \"the first incident\", null, 23, 200, \"Open\", 12, \'2012-06-18\')", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    public void testTaskList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<TaskManagementService>>(){});
        List<TaskManagementService> taskList = response.getBody();
        assertNotNull(taskList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(taskList.size(), tMServiceRepo.findAll().size());
        assertEquals(1, tMServiceRepo.findAll().size());
    }

    @Test
    @Order(2)
    public void testTaskById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<TaskManagementService>>(){});
        Optional<TaskManagementService> task = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(task, tMServiceRepo.findById(1L));
        assertEquals("the first incident", tMServiceRepo.findById(1L).get().getIncidentTitle());
    }

    @Test
    @Order(3)
    public void testTaskListByEmployeeId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/employee/23"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<TaskManagementService>>(){});
        List<TaskManagementService> taskList = response.getBody();
        assertNotNull(taskList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(taskList.size(), tMServiceRepo.findAllByEmployeeId(23).size());
        assertEquals(1, tMServiceRepo.findAllByEmployeeId(23).size());
    }

    @Test
    @Order(4)
    public void testTaskListByAccountId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<TaskManagementService>> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/account/12"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<TaskManagementService>>(){});
        List<TaskManagementService> taskList = response.getBody();
        assertNotNull(taskList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(taskList.size(), tMServiceRepo.findAllByAccountId(12).size());
        assertEquals(1, tMServiceRepo.findAllByAccountId(12).size());
    }

    @Test
    @Order(5)
    public void testCreateTask() throws JsonProcessingException {
        TaskManagementService task = new TaskManagementService(2L, "the second incident", null, 28, 300, "In review", 12, null);

        HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(task), headers);
        ResponseEntity<TaskManagementService> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks"), HttpMethod.POST, entity, TaskManagementService.class);
            
        assertEquals(response.getStatusCode(), HttpStatus.CREATED);
        TaskManagementService taskRes = Objects.requireNonNull(response.getBody());
        assertNotNull(taskRes);
        assertEquals(taskRes.getStatus(), tMServiceRepo.save(task).getStatus());
        assertEquals(taskRes.getIncidentTitle(), tMServiceRepo.save(task).getIncidentTitle());
        assertEquals(taskRes.getSeverity(), tMServiceRepo.save(task).getSeverity());
        
    }

    @Test
    @Order(6)
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
    @Order(7)
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
    @Order(8)
    public void testDeleteTask() {
        ResponseEntity<Void> response = restTemplate.exchange(
            (createURLWithPort() + "/tasks/1"), HttpMethod.DELETE, null, Void.class);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }
}
