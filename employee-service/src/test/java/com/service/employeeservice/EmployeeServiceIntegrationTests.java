package com.service.employeeservice;

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

import java.util.*;

import javax.sql.DataSource;

import com.zaxxer.hikari.HikariDataSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(locations = "classpath:application-test.properties")
@Testcontainers
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class EmployeeServiceIntegrationTests {
    

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
    @Order(1)
    @Sql(statements = "INSERT INTO employees (id, firstname, lastname, email, gender, business_unit, joined_date, terminated_date, location) values (1, \"Mary\", \"Elizabeth\", \"marye@gmail.com\", \"female\", \"Asset Management\", null, null, \"Seattle\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    void testEmployeeList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<EmployeeService>> response = restTemplate.exchange(
            (createURLWithPort() + "/employees"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<EmployeeService>>(){});
        List<EmployeeService> employeeList = response.getBody();
        assertNotNull(employeeList);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(employeeList.size(), empServiceRepo.findAll().size());
    }

    @Test
    @Order(2)
    @Sql(statements = "INSERT INTO employees (id, firstname, lastname, email, gender, business_unit, joined_date, terminated_date, location) values (2, \"Shelly\", \"Rose\", \"shel@gmail.com\", \"female\", \"Asset Management\", null, null, \"Seattle\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    void testEmployeeById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<EmployeeService>> response = restTemplate.exchange(
            (createURLWithPort() + "/employees/2"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<EmployeeService>>(){});
        Optional<EmployeeService> employee = response.getBody();

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(employee, empServiceRepo.findById(2L));
        assertEquals("Shelly", empServiceRepo.findById(2L).get().getFirstname());
    }

}