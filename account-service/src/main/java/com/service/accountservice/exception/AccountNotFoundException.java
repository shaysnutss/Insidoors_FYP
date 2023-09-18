package com.service.accountservice.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND) // 404 Error
public class AccountNotFoundException extends RuntimeException {

        public AccountNotFoundException(String email) {
            super(email + " not found");
        }

        public AccountNotFoundException(Long id) {
            super(id + " not found");
        }

    public AccountNotFoundException() {
        super("No Accounts Found");
    }

}
