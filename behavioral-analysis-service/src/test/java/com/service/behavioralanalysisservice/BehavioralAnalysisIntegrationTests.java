package com.service.behavioralanalysisservice;

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
public class BehavioralAnalysisIntegrationTests {
    
    @LocalServerPort
    private int port;

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

    private String createURLWithPort() {
        return "http://localhost:" + port + "/api/v1";
    }

    @Test
    @Sql(statements = "INSERT INTO behavioral_analysis_service (ba_id, employee_id, risk_rating, suspected_cases) values (1, 23, 120, 12)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM behavioral_analysis_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testBAList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<BehavioralAnalysisService>> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<BehavioralAnalysisService>>(){});
        List<BehavioralAnalysisService> bAList = response.getBody();
        assertNotNull(bAList);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(bAList.size(), bAServiceRepo.findAll().size());
    }

    @Test
    @Sql(statements = "INSERT INTO behavioral_analysis_service (ba_id, employee_id, risk_rating, suspected_cases) values (1, 23, 120, 12)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM behavioral_analysis_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testBAById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<BehavioralAnalysisService>> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<BehavioralAnalysisService>>(){});
        Optional<BehavioralAnalysisService> bAItem = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(bAItem, bAServiceRepo.findById(1L));
    }

    // test for getting risk rating by employee id
    
    // test for getting suspected cases by employee id

    @Test
    @Sql(statements = "DELETE FROM behavioral_analysis_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCreateBA() throws JsonProcessingException {
        BehavioralAnalysisService bA = new BehavioralAnalysisService(1L, 23, 120, 12);

        HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(bA), headers);
        ResponseEntity<BehavioralAnalysisService> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis"), HttpMethod.POST, entity, BehavioralAnalysisService.class);
            
        assertEquals(response.getStatusCode(), HttpStatus.CREATED);
        BehavioralAnalysisService baRes = Objects.requireNonNull(response.getBody());
        assertEquals(baRes.getRiskRating(), bAServiceRepo.save(bA).getRiskRating());
    }

    @Test
    @Sql(statements = "INSERT INTO behavioral_analysis_service (ba_id, employee_id, risk_rating, suspected_cases) values (1, 23, 120, 12)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM behavioral_analysis_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testUpdateRiskRating() throws JsonProcessingException {
        // Modify some properties of the initialBA object to simulate an update
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
    @Sql(statements = "INSERT INTO behavioral_analysis_service (ba_id, employee_id, risk_rating, suspected_cases) values (1, 23, 120, 12)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM behavioral_analysis_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testUpdateSuspectedCases() throws JsonProcessingException {
        // Modify some properties of the initialBA object to simulate an update
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
    @Sql(statements = "INSERT INTO behavioral_analysis_service (ba_id, employee_id, risk_rating, suspected_cases) values (1, 23, 120, 12)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    public void testDeleteBA() {
        ResponseEntity<Void> response = restTemplate.exchange(
            (createURLWithPort() + "/behavioralanalysis/1"), HttpMethod.DELETE, null, Void.class);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }

}
