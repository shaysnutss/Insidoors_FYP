package com.service.commentsservice;

import java.util.*;

import org.springframework.dao.EmptyResultDataAccessException;
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

@RestController
@RequiredArgsConstructor
@RequestMapping(path = "/api/v1")
public class CommentsServiceController {

    private final CommentsServiceRepository commentsServiceRepo;

    @GetMapping("/comments")
    public List<CommentsService> getAllComments(){
        return commentsServiceRepo.findAll();
    }

    @GetMapping("/comments/{id}") 
    public Optional<CommentsService> getCommentsById(@PathVariable Long id) {     
        return commentsServiceRepo.findById(id);
    }

    @GetMapping("/comments/taskManagement/{id}") 
    public List<CommentsService> getCommentsByTaskManagementId(@PathVariable int id) {     
        return commentsServiceRepo.findAllByTaskManagementId(id);
    }

    @GetMapping("/comments/account/{id}") 
    public List<CommentsService> getCommentsByAccountId(@PathVariable int id) {     
        return commentsServiceRepo.findAllByAccountId(id);
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/comments")
    public CommentsService addComment(@RequestBody CommentsService comment) {
        return commentsServiceRepo.save(comment);
    }

    @PutMapping("/comments/{id}")
    public CommentsService updateComment(@PathVariable Long id, @RequestBody CommentsService commentNew){
        Optional<CommentsService> comment = commentsServiceRepo.findById(id);

        if (comment == null) {
            throw new CommentNotFoundException(id);
        } 
        comment.get().setId(id);
        comment.get().setCommentDescription(commentNew.getCommentDescription());
        return commentsServiceRepo.save(comment.get());
        
    }
    
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
