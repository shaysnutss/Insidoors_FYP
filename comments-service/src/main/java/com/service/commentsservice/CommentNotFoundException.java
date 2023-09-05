package com.service.commentsservice;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND) 
public class CommentNotFoundException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public CommentNotFoundException(Long id) {
        super("Could not find the comment with id: " + id);
    }
    
}
