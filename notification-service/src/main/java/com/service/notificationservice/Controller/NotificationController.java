package com.service.notificationservice.Controller;


import com.service.notificationservice.Email.AmazonSES;
import com.service.notificationservice.Service.NotificationService;
import jakarta.mail.MessagingException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping(path = "/api/v1/notifications")
public class NotificationController {

    public final NotificationService notificationService;

    private final AmazonSES amazonSES;

    @Value("${ACCOUNT_API_BASE_URL}")
    private String accountApiBaseUrl;

    public NotificationController(NotificationService notificationService, AmazonSES amazonSES) {
        this.notificationService = notificationService;
        this.amazonSES = amazonSES;
    }


    @ResponseStatus(HttpStatus.CREATED)
    @PostMapping("/assignSOC/{id}")
    public ResponseEntity<Void> assignSOCNotification(@PathVariable(value = "id") Long accountID, @RequestBody Map<String, Object> taskData) throws MessagingException, IOException {

        // Extracting task details from request body
        int severity = (int) taskData.get("severity");
        String incidentTitle = (String) taskData.get("incidentTitle");

        // Sending GET request to account service to get account name
        String accountServiceUrl = accountApiBaseUrl + "/getAccountById/" + accountID;
        Map <String, Object> accountData = notificationService.getAccount(accountServiceUrl);

        if(accountData != null && ((amazonSES.sendEmail((String) accountData.get("name"), "insidoorsfyp@gmail.com",
                "[INSIDOORS] New Task Assigned" + " - Severity " + severity, severity, incidentTitle, (int) taskData.get("id") )) != null)){
            return ResponseEntity.ok().build();
        }

//        if(accountData != null && ((amazonSES.sendEmail((String) accountData.get("accountName"), (String) accountData.get("email"),
//                "New Task Assigned" + " - Severity" + severity, severity, incidentTitle)) != null)){
//            return ResponseEntity.ok().build();
//        }
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();


    }
    /*
    * int id = (int) taskData.get("id");
        String incidentTitle = (String) taskData.get("incidentTitle");
        LocalDateTime incidentTimestamp = LocalDateTime.parse((String) taskData.get("incidentTimestamp"));
        int employeeId = (int) taskData.get("employeeId");
        int severity = (int) taskData.get("severity");
        String status = (String) taskData.get("status");
        int accountId = (int) taskData.get("accountId");
        LocalDate dateAssigned = LocalDate.parse((String) taskData.get("dateAssigned"));*/





}
