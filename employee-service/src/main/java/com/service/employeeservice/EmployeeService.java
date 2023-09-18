package com.service.employeeservice;

import lombok.*;

import java.time.LocalDate;

import jakarta.persistence.*;

@Entity
@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode
@Table(name = "employee_service", schema = "employees")
public class EmployeeService {
    
    private @Id @Column(name="id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="firstname")
    private String firstname;

    @Column(name="lastname")
    private String lastname;

    @Column(name="email")
    private String email;

    @Column(name="gender")
    private String gender;

    @Column(name="business_unit")
    private String businessUnit;

    @Column(name="joined_date")
    private LocalDate joinedDate;

    @Column(name="terminated_date")
    private LocalDate terminatedDate;

    @Column(name="location")
    private String location;
}
