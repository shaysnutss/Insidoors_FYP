package com.service.employeeservice;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
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

import java.time.*;
import java.util.*;

import javax.sql.DataSource;

import com.zaxxer.hikari.HikariDataSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(locations = "classpath:application-test.properties")
@Testcontainers
public class EmployeeServiceIntegrationTests {
    
    @Autowired
    private static DataSource dataSource;

    @LocalServerPort
    private int port;

    @Container
    public static MySQLContainer<?> mySQLContainer = new MySQLContainer<>(DockerImageName.parse("mysql:8.0.26"))
        .withDatabaseName("employee")
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
    private EmployeeServiceRepository empServiceRepo;

    private static HttpHeaders headers;

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
    //@Sql(statements = "INSERT INTO employees (id, firstname, lastname, email, gender, business_unit, joined_date, terminated_date, location) values (23, \"Mary\", \"Elizabeth\", \"marye@gmail.com\", \"female\", \"Asset Management\", null, null, \"Seattle\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    //@Sql(statements = "DELETE FROM employees", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testEmployeeList() {
        List<EmployeeService> mockEmployees = new ArrayList<>(
            Arrays.asList(new EmployeeService(1L, "Mary", "Elizabeth", "marye@gmail.com", "female", "Asset Management", LocalDate.of(2017,Month.FEBRUARY,3), null, "Seattle"),
            new EmployeeService(2L, "Sean", "Henry", "seanh@gmail.com", "male", "Asset Management", LocalDate.of(2017,Month.FEBRUARY,3), null, "London")));
    
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<EmployeeService>> response = restTemplate.exchange(
            (createURLWithPort() + "/employees"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<EmployeeService>>(){});
        List<EmployeeService> employeeList = response.getBody();
        assertNotNull(employeeList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(employeeList.size(), empServiceRepo.findAll().size());
    }

    @Test
    //@Sql(statements = "INSERT INTO employees (id, firstname, lastname, email, gender, business_unit, joined_date, terminated_date, location) values (23, \"Mary\", \"Elizabeth\", \"marye@gmail.com\", \"female\", \"Asset Management\", null, null, \"Seattle\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    //@Sql(statements = "DELETE FROM employees", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testEmployeeById() {
        EmployeeService mockEmployee = new EmployeeService(1L, "Mary", "Elizabeth", "marye@gmail.com", "female", "Asset Management", LocalDate.of(2017,Month.FEBRUARY,3), null, "Seattle");

        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<EmployeeService>> response = restTemplate.exchange(
            (createURLWithPort() + "/employees/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<EmployeeService>>(){});
        Optional<EmployeeService> employee = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(employee, empServiceRepo.findById(1L));
    }

}