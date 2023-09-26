package com.service.commentsservice;

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
public class CommentsServiceIntegrationTest {
    
    @LocalServerPort
    private int port;

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

    private String createURLWithPort() {
        return "http://localhost:" + port + "/api/v1";
    }

    @Test
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM comments_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCommentsList() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();
        assertNotNull(comments);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comments.size(), commentsServiceRepo.findAll().size());
    }

    @Test
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM comments_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCommentsById() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<Optional<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Optional<CommentsService>>(){});
        Optional<CommentsService> comment = response.getBody();

        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comment, commentsServiceRepo.findById(1L));
    }

    @Test
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM comments_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCommentsListByTaskManagementId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/taskManagement/12"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();
        assertNotNull(comments);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comments.size(), commentsServiceRepo.findAllByTaskManagementId(12).size());
    }

    @Test
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM comments_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCommentsListByAccountId() {
        HttpEntity<String> entity = new HttpEntity<>(null, headers);
        ResponseEntity<List<CommentsService>> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/account/8"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<CommentsService>>(){});
        List<CommentsService> comments = response.getBody();
        assertNotNull(comments);
        assertEquals(response.getStatusCode(), HttpStatus.OK);
        assertEquals(comments.size(), commentsServiceRepo.findAllByAccountId(8).size());
    }

    @Test
    @Sql(statements = "DELETE FROM comments_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
    public void testCreateBA() throws JsonProcessingException {
        CommentsService comment = new CommentsService(1L, "the first comment", 12, 8);

        HttpEntity<String> entity = new HttpEntity<>(objectMapper.writeValueAsString(comment), headers);
        ResponseEntity<CommentsService> response = restTemplate.exchange(
            (createURLWithPort() + "/comments"), HttpMethod.POST, entity, CommentsService.class);
            
        assertEquals(response.getStatusCode(), HttpStatus.CREATED);
        CommentsService commentRes = Objects.requireNonNull(response.getBody());
        assertEquals(commentRes.getCommentDescription(), commentsServiceRepo.save(comment).getCommentDescription());
    }

    @Test
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    @Sql(statements = "DELETE FROM comments_service", executionPhase = Sql.ExecutionPhase.AFTER_TEST_METHOD)
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
    @Sql(statements = "INSERT INTO comments_service (comment_id, comment_description, task_management_id, account_id) values (1, \"the first comment\", 12, 8)", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
    public void testDeleteComment() {
        ResponseEntity<Void> response = restTemplate.exchange(
            (createURLWithPort() + "/comments/1"), HttpMethod.DELETE, null, Void.class);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }
}
