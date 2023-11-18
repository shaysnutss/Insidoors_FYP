package com.service.commentsservice;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

@RestController
@CrossOrigin(maxAge = 3600)
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class CommentsServiceController {

    private final CommentsServiceRepository commentsServiceRepo;

    @CrossOrigin(origins = "http://localhost:30008")   
    @GetMapping("/comments")
    public ResponseEntity<List<CommentsService>> getAllComments() {
        List<CommentsService> comments = commentsServiceRepo.findAll();

        if (comments.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } else {
            return ResponseEntity.ok(comments);
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")   
    @GetMapping("/comments/{id}") 
    public ResponseEntity<CommentsService> getCommentsById(@PathVariable Long id) {
        Optional<CommentsService> commentOptional = commentsServiceRepo.findById(id);

        if (commentOptional.isPresent()) {
            return ResponseEntity.ok(commentOptional.get());
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")   
    @GetMapping("/comments/taskManagement/{id}") 
    public ResponseEntity<List<CommentsService>> getCommentsByTaskManagementId(@PathVariable int id) {
        List<CommentsService> comments = commentsServiceRepo.findAllByTaskManagementId(id);

        if (!comments.isEmpty()) {
            return ResponseEntity.ok(comments);
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")   
    @GetMapping("/comments/account/{id}") 
    public ResponseEntity<List<CommentsService>> getCommentsByAccountId(@PathVariable int id) {
        List<CommentsService> comments = commentsServiceRepo.findAllByAccountId(id);

        if (!comments.isEmpty()) {
            return ResponseEntity.ok(comments);
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/comments")
    public CommentsService addComment(@RequestBody String comment) throws JsonMappingException, JsonProcessingException {
        CommentsService commentsService = new CommentsService();
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode jsonNode = objectMapper.readTree(comment);

        commentsService.setCommentDescription(jsonNode.get("commentDescription").asText());
        commentsService.setAccountId(jsonNode.get("accountId").asInt());
        commentsService.setTaskManagementId(jsonNode.get("taskManagementId").asInt());
        return commentsServiceRepo.save(commentsService);
    }

    @CrossOrigin(origins = "http://localhost:30008")   
    @PutMapping("/comments/{id}")
    public ResponseEntity<CommentsService> updateComment(@PathVariable Long id, @RequestBody String commentNew){
        CommentsService comment = commentsServiceRepo.findById(id).get();

        try {
            if (comment == null) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
            }

            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonNode = objectMapper.readTree(commentNew);
            comment.setId(id);
            String commentDescription = jsonNode.get("commentDescription").asText();
            comment.setCommentDescription(commentDescription);

        } catch (JsonMappingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        }

        commentsServiceRepo.save(comment);
        return ResponseEntity.ok(comment);
        
    }
    
    @CrossOrigin(origins = "http://localhost:30008")
    @DeleteMapping("/comments/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteComment(@PathVariable Long id){
        try {
            commentsServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            throw new CommentNotFoundException(id);
        }
    }
}
