package com.service.employeeservice;

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
public class EmployeeServiceIntegrationTests {
    
    @LocalServerPort
    private int port;

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

    private String createURLWithPort() {
        return "http://localhost:" + port + "/api/v1";
    }

    @Test
    @Sql(statements = "INSERT INTO employees (id, firstname, lastname, email, gender, business_unit, joined_date, terminated_date, location) values (23, \"Mary\", \"Elizabeth\", \"marye@gmail.com\", \"female\", \"Asset Management\", null, null, \"Seattle\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM employees", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testEmployeeList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<EmployeeService>> response = restTemplate.exchange(
            (createURLWithPort() + "/employees"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<EmployeeService>>(){});
        List<EmployeeService> employeeList = response.getBody();
        assertNotNull(employeeList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(employeeList.size(), empServiceRepo.findAll().size());
    }

    @Test
    @Sql(statements = "INSERT INTO employees (id, firstname, lastname, email, gender, business_unit, joined_date, terminated_date, location) values (23, \"Mary\", \"Elizabeth\", \"marye@gmail.com\", \"female\", \"Asset Management\", null, null, \"Seattle\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM employees", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testEmployeeById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<EmployeeService>> response = restTemplate.exchange(
            (createURLWithPort() + "/employees/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<EmployeeService>>(){});
        Optional<EmployeeService> employee = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(employee, empServiceRepo.findById(1L));
    }

}
