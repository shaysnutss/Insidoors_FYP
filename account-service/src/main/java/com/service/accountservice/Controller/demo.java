package com.service.accountservice.Controller;

import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
@CrossOrigin(origins = "http://localhost:3000")
@RequestMapping("/api/v1/demo-controller")
public class demo {
    @GetMapping
    public ResponseEntity<String> sayHello() {
        return ResponseEntity.ok("access granted");
    }
}
