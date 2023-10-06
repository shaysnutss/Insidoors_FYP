package com.service.commentsservice;

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
public class CommentsServiceIntegrationTest {

    private static DataSource dataSource;
    
    @LocalServerPort
    private int port;

    @Container
    public static MySQLContainer<?> mySQLContainer = new MySQLContainer<>(DockerImageName.parse("mysql:8.0.26"))
        .withDatabaseName("comments")
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
    private CommentsServiceRepository commentsServiceRepo;

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
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    public void testCommentsList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();
        assertNotNull(comments);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comments.size(), commentsServiceRepo.findAll().size());
        assertEquals(1, commentsServiceRepo.findAll().size());
    }

    @Test
    @Order(2)
    public void testCommentsById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<CommentsService>>(){});
        Optional<CommentsService> comment = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comment, commentsServiceRepo.findById(1L));
        assertEquals("the first comment", commentsServiceRepo.findById(1L).get().getCommentDescription());
    }

    @Test
    @Order(3)
    public void testCommentsListByTaskManagementId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/taskManagement/12"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();
        assertNotNull(comments);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comments.size(), commentsServiceRepo.findAllByTaskManagementId(12).size());
        assertEquals(1, commentsServiceRepo.findAllByTaskManagementId(12).size());
    }

    @Test
    @Order(4)
    public void testCommentsListByAccountId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/account/8"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();
        assertNotNull(comments);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comments.size(), commentsServiceRepo.findAllByAccountId(8).size());
        assertEquals(1, commentsServiceRepo.findAllByAccountId(8).size());
    }

    @Test
    @Order(5)
    public void testCommentsListByWrongAccountId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/account/300"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.NOT_FOUND);
        assertEquals(0, commentsServiceRepo.findAllByAccountId(300).size());
    }

    @Test
    @Order(6)
    public void testCreateBA() throws JsonProcessingException {
        CommentsService comment = new CommentsService(2L, "the second comment", 13, 9);

        HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(comment), headers);
        ResponseEntity<CommentsService> response = restTemplate.exchange(
            (createURLWithPort() + "/comments"), HttpMethod.POST, entity, CommentsService.class);
            
        assertEquals(response.getStatusCode(), HttpStatus.CREATED);
        CommentsService commentRes = Objects.requireNonNull(response.getBody());
        assertEquals(commentRes.getCommentDescription(), commentsServiceRepo.save(comment).getCommentDescription());
        assertEquals("the second comment", commentsServiceRepo.save(comment).getCommentDescription());
    }

    @Test
    @Order(7)
    public void testUpdateComment() throws JsonProcessingException {
        CommentsService commentNew = new CommentsService();
        commentNew.setCommentDescription("updated");
        commentNew.setId(1L);

        HttpEntity<CommentsService> entity = new HttpEntity<>(commentNew, null);

        ResponseEntity<CommentsService> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/1"), HttpMethod.PUT, entity, CommentsService.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals("updated", response.getBody().getCommentDescription());
    }


    @Test
    @Order(8)
    public void testDeleteComment() {
        ResponseEntity<Void> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/1"), HttpMethod.DELETE, null, Void.class);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }
}
