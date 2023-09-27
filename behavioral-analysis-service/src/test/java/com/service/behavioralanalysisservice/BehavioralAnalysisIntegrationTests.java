package com.service.behavioralanalysisservice;

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
public class BehavioralAnalysisIntegrationTests {

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
    private BehavioralAnalysisServiceRepository bAServiceRepo;

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
    @Sql(statements = "INSERT INTO behavioral_analysis_service (ba_id, employee_id, risk_rating, suspected_cases) values (1, 23, 120, 12)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    public void testBAList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<BehavioralAnalysisService>> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<BehavioralAnalysisService>>(){});
        List<BehavioralAnalysisService> bAList = response.getBody();
        assertNotNull(bAList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(bAList.size(), bAServiceRepo.findAll().size());
        assertEquals(1, bAServiceRepo.findAll().size());
    }

    @Test
    @Order(2)
    public void testBAById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<BehavioralAnalysisService>> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<BehavioralAnalysisService>>(){});
        Optional<BehavioralAnalysisService> bAItem = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(bAItem, bAServiceRepo.findById(1L));
        assertEquals(120, bAServiceRepo.findById(1L).get().getRiskRating());
    }

    @Test
    @Order(3)
    public void testCreateBA() throws JsonProcessingException {
        BehavioralAnalysisService bA = new BehavioralAnalysisService(2L, 24, 130, 14);

        HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(bA), headers);
        ResponseEntity<BehavioralAnalysisService> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis"), HttpMethod.POST, entity, BehavioralAnalysisService.class);
            
        assertEquals(response.getStatusCode(), HttpStatus.CREATED);
        BehavioralAnalysisService baRes = Objects.requireNonNull(response.getBody());
        assertEquals(baRes.getRiskRating(), bAServiceRepo.save(bA).getRiskRating());
        assertEquals(130, bAServiceRepo.save(bA).getRiskRating());
    }

    @Test
    @Order(4)
    public void testUpdateRiskRating() throws JsonProcessingException {
        BehavioralAnalysisService bA = new BehavioralAnalysisService();
        bA.setRiskRating(30);
        bA.setId(1L);
        bA.setEmployeeId(23);

        ObjectMapper mapper = new ObjectMapper();
        String requestBody = mapper.writeValueAsString(bA);

        HttpEntity<String> entity = new HttpEntity<>(requestBody, null);

        ResponseEntity<BehavioralAnalysisService> response = restTemplate.exchange(
            (createURLWithPort() + "/updateRiskRating/23"), HttpMethod.PUT, entity, BehavioralAnalysisService.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals(150, response.getBody().getRiskRating());
    }

    @Test
    @Order(5)
    public void testUpdateSuspectedCases() throws JsonProcessingException {
        BehavioralAnalysisService bA = new BehavioralAnalysisService();
        bA.setSuspectedCases(30);
        bA.setId(1L);
        bA.setEmployeeId(23);

        ObjectMapper mapper = new ObjectMapper();
        String requestBody = mapper.writeValueAsString(bA);

        HttpEntity<String> entity = new HttpEntity<>(requestBody, null);

        ResponseEntity<BehavioralAnalysisService> response = restTemplate.exchange(
            (createURLWithPort() + "/updateSuspectedCases/23"), HttpMethod.PUT, entity, BehavioralAnalysisService.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals(30, response.getBody().getSuspectedCases());
    }

    @Test
    @Order(6)
    public void testDeleteBA() {
        ResponseEntity<Void> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis/1"), HttpMethod.DELETE, null, Void.class);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }

}
