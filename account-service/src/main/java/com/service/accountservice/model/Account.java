package com.service.accountservice.model;



import jakarta.validation.constraints.Email;
import lombok.*;
import jakarta.persistence.*;

@Entity
@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode
@Table(name = "account_service", schema = "account")
public class Account {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id")
    private Long id;

    @Column(name = "name", length = 50, nullable = false)
    private String name;

    @Column(name = "email", length = 70, unique = true ,nullable = false)
    @Email
    private String email;

    @Column(name = "password", length = 70, nullable = false)
    private String password;

    @Column(name = "department", length = 50, nullable = false)
    private String department;

    @Column(name = "role", length = 50, nullable = false)
    private String role;

    @Column(name = "number", length = 50)
    private int number;

}
