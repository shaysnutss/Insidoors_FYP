package com.service.commentsservice;

import lombok.*;
import jakarta.persistence.*;

@Entity
@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode
@Table(name = "comments_service", schema = "comments")
public class CommentsService {
    
    private @Id @Column(name="comment_id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="comment_description")
    private String commentDescription;
    
    @Column(name="task_management_id")
    private Long taskManagementId;

    @Column(name="account_id")
    private Long accountId;

    public CommentsService(String commentDescription, Long taskManagementId, Long accountId) {
        this.commentDescription = commentDescription;
        this.taskManagementId = taskManagementId;
        this.accountId = accountId;
    }

}
