package com.service.accountservice.repository;

import java.util.*;

import com.service.accountservice.model.Account;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AccountServiceRepository extends JpaRepository<Account, Long> {

    // find an account by the email as email is unique
    Account findByEmail(String email);

}
