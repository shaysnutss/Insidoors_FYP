package com.service.accountservice.Controller;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.service.accountservice.Service.accountService;
import com.service.accountservice.config.JwtService;
import com.service.accountservice.repository.AccountServiceRepository;
import com.service.accountservice.model.Account;

import jakarta.servlet.http.HttpServletRequest;

@RestController
@CrossOrigin(origins = "http://localhost:30008")
@RequestMapping(path = "/api/v1/accountByTasks")
@RequiredArgsConstructor
public class TaskCompositeController {

    private final accountService accountService;
    private final JwtService jwtService;
    private final AccountServiceRepository repository;


    @Value("${TASK_COMP_API_BASE_URL}")
    private String taskMCompApiBaseUrl;

    @Value("${COMMENTS_API_BASE_URL}")
    private String commentsApiBaseUrl;

    @Value("${BA_COMPOSITE_API_BASE_URL}")
    private String baCompositeApiBaseURL;

    

    @GetMapping("/viewAllTasks")
    public ResponseEntity<?> viewAllTasks(){
        return accountService.getMethod(taskMCompApiBaseUrl + "/viewAllTasks");
    }

    @GetMapping("/viewAllTasksByAccountId/{id}")
    public ResponseEntity<?> viewAllTasksByAccountId(@PathVariable int id){
        return accountService.getMethod(taskMCompApiBaseUrl + "/viewAllTasksByAccountId/" + id);
    }

    @GetMapping("/viewAllCommentsByTaskId/{id}")
    public ResponseEntity<?> viewAllCommentsByTaskId(@PathVariable int id){
        return accountService.getMethod(taskMCompApiBaseUrl + "/viewAllCommentsByTaskId/" + id);
    }

    @GetMapping("/listSOCs")
    public ResponseEntity<?> listSOCs(){
        return accountService.getMethod(taskMCompApiBaseUrl + "/listSOCs");
    }

    @PutMapping("/assignSOC/{id}")
    public ResponseEntity<?> assignSOCs(@PathVariable Long id, @RequestBody String requestBody){
        return accountService.putMethod(taskMCompApiBaseUrl + "/assignSOC/" + id, requestBody);
    }

    @PutMapping("/changeStatus/{id}")
    public ResponseEntity<?> changeStatus(@PathVariable Long id, @RequestBody String requestBody){
        return accountService.putMethod(taskMCompApiBaseUrl + "/changeStatus/" + id, requestBody);
    }

    @PutMapping("/closeCase/{id}")
    public ResponseEntity<?> closeCase(@PathVariable Long id, @RequestBody String requestBody){
        return accountService.putMethod(taskMCompApiBaseUrl + "/closeCase/" + id, requestBody);
    }

    @GetMapping("/viewAllTasksByEmployeeId/{id}")
    public ResponseEntity<?> viewAllTasksByEmployeeId(@PathVariable int id){
        return accountService.getMethod(taskMCompApiBaseUrl + "/viewAllTasksByEmployeeId/" + id);
    }

    @GetMapping("/viewName")
    public ResponseEntity<?> viewName(HttpServletRequest request){
        String authHeader = request.getHeader("Authorization");
        if(authHeader == null || !authHeader.startsWith("Bearer ")){
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body("No bearer token provided. This web app uses JWT Authentication");
        }

        String jwt = authHeader.substring(7);
        String userEmail = jwtService.extractUsername(jwt);
        Account accountToReturn = repository.findByEmail(userEmail);
        return ResponseEntity.ok(accountToReturn.getName());
        
    }

    @GetMapping("/getAccount")
    public ResponseEntity<?> getAccount(HttpServletRequest request){
        String authHeader = request.getHeader("Authorization");
        if(authHeader == null || !authHeader.startsWith("Bearer ")){
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body("No bearer token provided. This web app uses JWT Authentication");
        }

        String jwt = authHeader.substring(7);
        String userEmail = jwtService.extractUsername(jwt);
        Account accountToReturn = repository.findByEmail(userEmail);
        return ResponseEntity.ok(accountToReturn);
        
    }

}
