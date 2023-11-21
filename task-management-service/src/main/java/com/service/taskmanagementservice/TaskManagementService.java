package com.service.taskmanagementservice;

import lombok.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Date;

import jakarta.persistence.*;

@Entity
@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode
@Table(name = "task_management_service", schema = "task_management")
public class TaskManagementService {
    
    private @Id @Column(name="tm_id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="incident_title")
    private String incidentTitle;

    @Column(name="incident_desc")
    private String incidentDesc;

    @Column(name="incident_timestamp")
    private String incidentTimestamp;
    
    @Column(name="employee_id")
    private int employeeId;
    
    @Column(name="severity")
    private int severity;

    @Column(name="status")
    private String status;

    @Column(name="account_id")
    private int accountId;

    @Column(name="true_positive")
    private Boolean truePositive = false;

}
