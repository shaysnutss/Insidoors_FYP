package com.service.commentsservice;

import java.util.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.HttpStatus;

@RestController
public class CommentsServiceController {

    @Autowired
    private CommentsServiceRepository commentsServiceRepo;

    @GetMapping("/comments")
    public List<CommentsService> getAllComments(){
        return commentsServiceRepo.findAll();
    }

    @GetMapping("/comments/{id}") 
    public Optional<CommentsService> getCommentsById(@PathVariable Long id) {     
        return commentsServiceRepo.findById(id);
    }

    @GetMapping("/comments/taskManagement/{id}") 
    public List<CommentsService> getCommentsByTaskManagementId(@PathVariable Long id) {     
        return commentsServiceRepo.findByTaskManagementId(id);
    }

    @GetMapping("/comments/account/{id}") 
    public List<CommentsService> getCommentsByAccountId(@PathVariable Long id) {     
        return commentsServiceRepo.findByAccountId(id);
    }

    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/comments")
    public CommentsService addComment(@RequestBody CommentsService comment) {
        return commentsServiceRepo.save(comment);
    }

    @PutMapping("/comments/{id}")
    public CommentsService updateComment(@PathVariable Long id, @RequestBody CommentsService commentNew){
        CommentsService comment = commentsServiceRepo.findById(id).orElse(null);

        if (comment == null) {
            throw new CommentNotFoundException(id);
        } 
        comment.setId(id);
        comment.setAccountId(commentNew.getAccountId());
        comment.setCommentDescription(commentNew.getCommentDescription());
        comment.setTaskManagementId(commentNew.getTaskManagementId());
        return commentsServiceRepo.save(comment);
        
    }
    
    @DeleteMapping("/comments/{id}")
    public void deleteNews(@PathVariable Long id){
        //newsService.deleteNews(id);
        try {
            commentsServiceRepo.deleteById(id);
        } catch(EmptyResultDataAccessException e) {
            throw new CommentNotFoundException(id);
        }
    }
}
