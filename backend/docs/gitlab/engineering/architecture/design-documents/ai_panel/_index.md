---
title: "AI Panel"
status: ongoing
creation-date: "2025-11-21"
authors: [ "@fredericcaplette"]
coach: ["@ntepluhina"]
approvers: [ ]
owning-stage: "ai"
participating-stages: ["plan"]
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

The AI panel is a side panel component that provides a unified interface for various AI-powered features in GitLab. It serves as a container that dynamically loads different sub-applications while maintaining a consistent navigation and layout structure.

## Terminology

### AI panel

The main container component (`ai_panel.vue`) that manages the overall panel state, navigation, and content rendering.

### Navigation rail

The vertical navigation component (`navigation_rail.vue`) that provides tab-based navigation for the AI panel. It renders icon buttons for different AI features (chat, history, sessions, suggestions) and handles tab switching, keyboard shortcuts, and disabled states. The navigation rail adapts its layout responsively (vertical on desktop, horizontal on mobile) and communicates with the AI panel through events to maintain loose coupling.

### Content container

The rendering layer component (`content_container.vue`) that sits between the AI panel infrastructure and sub-applications. It dynamically renders the active tab's component, manages the panel header (title, back button, collapse/maximize controls), and relays events between sub-applications and the panel while maintaining the props passthrough pattern for context data.

## Motivation

As GitLab's AI capabilities expandand during the FY26, it became clear that AI could not simply be added inside the application, but that we should invest in a way for AI interactions to follow the user around so that they could always benefit for the productivity boost they could get from delegating work to AI. The solution for this was to build a UI component as a Side Panel, which would be responsive across all of GitLab's platform.

Then separately, we wanted the AI teams to start integrating their work inside the panel. This separation meant that we needed a flexible architecture that could accommodate multiple AI features (chat, agents, history, sessions) without tightly coupling the panel UI and Ux behhaviors to specific implementations and allow quick iterations and turn around for both teams. By doing so, we have built and enforced a system that increase the team confidence to iterate fast by creating clear separation between apps to avoid one team's work unintentionally breaking someone else's work.

## Goals

- Provide a reusable panel container that remains agnostic about its sub-applications and work on all pages across GitLab
- Enable dynamic content switching based on user navigation
- Maintain consistent panel behavior (open/close, maximize, navigation) across all AI features and pages
- Support both classic and agentic chat modes without duplicating panel logic

## High-level architecture problems to solve

### Configuration Injection Pattern

The AI panel uses Vue's `provide/inject` mechanism to pass static data coming from rails and injecting directly into the app that require this data without explicit passthrough the AI Panel, Navigation Rail and Content container. For example, to configure chat components in (`init_duo_panel.js`), the `chatConfiguration` object is created during mount and injected into the component tree, containing:

- Component references for both agentic and classic chat modes
- Titles for each mode
- Default props shared across all chat instances

This pattern enables the panel to remain agnostic about sub-applications implementations while allowing the initialization layer to determine which components are available based on feature flags and user permissions.

### Vue Router Integration

The panel integrates with Vue Router to enable deep linking and navigation within AI features. The router is created at initialization (`createRouter`) and shared across the panel and its sub-applications, particularly for the agent sessions feature. Duo Chat is not using this router because it still relies on its own internal `mode` switching logic, which in time we should deprecate and instead support a single component per route.

This shared router allows:

- Direct URL access to specific panel states (e.g., `/agent-sessions/`)
- Browser back/forward navigation within the panel
- Route-based component rendering without coupling the panel to specific routes

The `currentTabComponent.initialRoute` property acts as a bridge between tab-based navigation and router-based navigation, enabling the panel to delegate routing concerns to sub-applications. Any sub-application that is not duo chat (which owns the root path "/"), should define its initial route. In time, we could iterate to make this dynamically loaded from the initial router configuration.

### Component Agnosticism

The `ai_panel.vue`, `content_container` and `navigation_rails` are all intentionally designed to be agnostic about the specific sub-applications they renders. This is achieved through:

**Dynamic component loading**: The panel uses a configuration-based approach where applications specifics that need to come from rails are passed through the `provide/inject` parttern as described above. This allows different implementations (classic vs agentic) to be swapped without modifying the panel itself.

**Tab-based content mapping**: The `currentTabComponent` computed property maps tab identifiers to their respective components and props, acting as a routing layer that keeps the panel decoupled from specific feature implementations.

**Props passthrough pattern**: The panel accepts generic props (`userId`, `projectId`, `namespaceId`, etc.) and passes them down to child components via spread operators, ensuring it doesn't need to know about specific sub-app requirements.

**Event-driven communication**: The panel communicates with sub-applications through events (`close-panel`, `go-back`, `switch-to-active-tab`) that have been named as generic action as opposed to having events that are exclusively used for a single app (`close-chat`, `switch-to-chat-history`, etc). It is essential to keep the ai_panel layer only a UI element without business logic as it enables teams to develop and iterate on individual AI features independently while the panel provides consistent infrastructure for navigation, state management, and layout.

### Container Pattern for Navigation Rail Components

As AI features grow in complexity, some navigation rail items require more than simple button interactions. Components like agent selectors and recent chats need their own API calls, state management, and non-trivial logic while remaining clearly part of a specific app's experience (e.g., Duo Chat).

To handle these "navigation rail mini-apps" while maintaining architectural boundaries, we use a **container pattern** that creates a clear separation between panel-level concerns and app-specific navigation logic.

#### When to use containers

Create a container component when navigation rail items for your app require:

- **API calls and data fetching** - The component needs to query GraphQL or REST endpoints
- **State management** - The component manages its own state beyond simple UI toggles
- **Complex interactions** - The component has non-trivial logic, event handling, or user workflows
- **Multiple related items** - Several navigation buttons share state or logic for the same app

Examples that warrant containers:

- Agent selector with API-driven agent list and selection state
- Recent chats feature with chat history fetching and management
- Notification center with unread counts and real-time updates

Examples that don't need containers:

- Simple navigation buttons that only emit tab-switching events
- Static icons with tooltips
- Dividers or visual separators

#### Container responsibilities

A navigation rail container component should:

**Own app-specific state**

- Manage state relevant only to its app (e.g., selected agent, active chat mode, chat availability)
- Keep this state isolated from the navigation rail and other apps
- Update state through Apollo cache mutations or component data as appropriate

**Handle app-specific events and logic**

- Process user interactions specific to the app's navigation items
- Make API calls needed for navigation features
- Implement business logic for enabling/disabling navigation options

**Group related navigation items**

- Render all navigation buttons/components for a single app
- Maintain consistent styling and layout with the navigation rail
- Provide a single entry point for the app's navigation presence

**Provide clear boundaries**

- Expose a simple interface to the navigation rail (props in, events out)
- Encapsulate complexity so the navigation rail remains simple
- Enable future navigation features to be added without modifying the rail

#### Contract with navigation rail

The container communicates with the navigation rail through a minimal interface:

**Props received from navigation rail**:

- `activeTab` - Current active tab identifier
- `isExpanded` - Panel expansion state
- App-specific configuration (e.g., `isChatDisabled`, `chatDisabledTooltip`)
- Context data (e.g., `projectId`, `namespaceId`)

**Events emitted to navigation rail**:

- `handleTabToggle` - Request tab navigation (panel-level concern)
- App-specific actions that require panel updates (e.g., `new-chat`, `newChatError`)

The navigation rail should not:

- Know about the container's internal state or logic
- Make assumptions about how the container implements its features
- Directly manipulate container components, their data or manually call methods

#### Benefits

This pattern provides:

- **Clear separation of concerns** - Panel-level logic stays in the navigation rail, app-specific logic lives in containers
- **Easier maintenance** - Changes to one app's navigation don't affect other apps or the rail
- **Better scalability** - New features can be added to containers without modifying the rail
- **Improved testability** - Containers can be unit tested in isolation from the navigation rail

### Navigation rail

The `navigation_rail.vue` component serves two purposes: providing navigation between different AI features and customizing the overall AI panel experience. From an engineering perspective, this component should only handle concerns that affect multiple applications, not features specific to a single app like Duo Chat.

#### Current architecture limitation

Currently, all navigation items (new chat, current chat, history, sessions, suggestions) are rendered at the same level within the navigation rail. As the application has grown, this flat structure has revealed a design issue: Duo Chat-specific navigation items (`new_chat`, `current_chat`, `history`) are mixed with panel-level navigation (sessions, suggestions), making it difficult to manage Duo Chat-specific state and logic.

#### Applying the container pattern to Duo Chat

Following the **Container Pattern for Navigation Rail Components** described above, we should introduce a new `duo_chat_navigation_container.vue` component that encapsulates all Duo Chat-related navigation items.

This container would implement the pattern by:

**Owning Duo Chat navigation state**

- Managing active chat mode (classic vs. agentic)
- Tracking chat availability and disabled states
- Handling agent selection state (for agentic mode)

**Grouping related navigation items**

- `new_chat` button with agent selector functionality
- `current_chat` button for active conversation
- `history` button for chat history

**Maintaining the navigation rail contract**

- Receiving props: `activeTab`, `isExpanded`, `isChatDisabled`, `chatDisabledTooltip`, `projectId`, `namespaceId`
- Emitting events: `handleTabToggle`, `new-chat`, `newChatError`

This creates a clearer component hierarchy where:

- **Navigation rail** handles panel-level concerns (layout, sessions, suggestions)
- **Duo Chat navigation container** handles chat-specific concerns (chat modes, history, new chat)
- **Individual navigation buttons** remain simple, presentational components

Logic that needs to be shared across the entire panel can be bubbled up to the navigation rail or AI panel component, ensuring each component operates at the appropriate level of abstraction.

**Code example**

Here's how we would refactor `navigation_rail.vue` to extract Duo Chat navigation into its own container:

```vue
<!-- navigation_rail.vue - Simplified to focus on panel-level concerns -->
<template>
  <div
    class="gl-ml-3 gl-flex gl-items-center gl-gap-3..."
    role="tablist"
    aria-orientation="vertical"
  >
    <!-- Duo Chat navigation container handles all chat-related items -->
    <duo-chat-navigation-container
      :active-tab="activeTab"
      :is-expanded="isExpanded"
      :is-chat-disabled="isChatDisabled"
      :chat-disabled-tooltip="chatDisabledTooltip"
      :project-id="projectId"
      :namespace-id="namespaceId"
      @handleTabToggle="toggleTab"
      @new-chat="handleNewChat"
      @newChatError="handleNewChatError"
    />

    <!-- Panel-level navigation items remain in the rail -->
    <template v-if="showSessionsButton">
      <div class="gl-my-3 gl-h-5 gl-w-1..." name="divider"></div>
      <gl-button
        v-gl-tooltip.left
        icon="session-ai"
        :class="['ai-nav-icon', { 'ai-nav-icon-active': activeTab === 'sessions' }]"
        @click="toggleTab('sessions')"
      />
    </template>

    <gl-button
      v-if="showSuggestionsTab"
      v-gl-tooltip.left
      icon="suggestion-ai"
      class="!gl-rounded-lg max-lg:gl-ml-auto lg:gl-mt-auto"
      :class="['ai-nav-icon', { 'ai-nav-icon-active': activeTab === 'suggestions' }]"
      @click="toggleTab('suggestions')"
    />
  </div>
</template>

<script>
import DuoChatNavigationContainer from './duo_chat_navigation_container.vue';

export default {
  name: 'NavigationRail',
  components: {
    DuoChatNavigationContainer,
  },
  // ... rest of component logic
};
</script>
```

```vue
<!-- duo_chat_navigation_container.vue - New component for chat navigation -->
<template>
  <div class="gl-flex gl-items-center gl-gap-3 lg:gl-flex-col">
    <new-chat-button
      :project-id="projectId"
      :namespace-id="namespaceId"
      :active-tab="activeTab"
      :is-expanded="isExpanded"
      :is-chat-disabled="isChatDisabled"
      :is-agent-select-enabled="isAgenticMode"
      :chat-disabled-tooltip="chatDisabledTooltip"
      @new-chat="$emit('new-chat', $event)"
      @toggleTab="$emit('handleTabToggle', $event)"
      @newChatError="$emit('newChatError', $event)"
    />

    <gl-button
      v-gl-tooltip.left="{ title: formattedDuoShortcutTooltip, html: true }"
      icon="duo-chat"
      :class="['ai-nav-icon', { 'ai-nav-icon-active': activeTab === 'chat' }]"
      @click="$emit('handleTabToggle', 'chat')"
    />

    <gl-button
      v-gl-tooltip.left
      icon="history"
      :class="['ai-nav-icon', { 'ai-nav-icon-active': activeTab === 'history' }]"
      @click="$emit('handleTabToggle', 'history')"
    />
  </div>
</template>

<script>
import { duoChatGlobalState } from '~/super_sidebar/constants';
import { CHAT_MODES } from 'ee/ai/tanuki_bot/constants';
import NewChatButton from './new_chat_button.vue';

export default {
  name: 'DuoChatNavigationContainer',
  components: {
    NewChatButton,
  },
  props: {
    activeTab: { type: String, default: null },
    isExpanded: { type: Boolean, default: true },
    isChatDisabled: { type: Boolean, default: false },
    chatDisabledTooltip: { type: String, default: '' },
    projectId: { type: String, default: null },
    namespaceId: { type: String, default: null },
  },
  data() {
    return {
      duoChatGlobalState,
    };
  },
  computed: {
    isAgenticMode() {
      return this.duoChatGlobalState.chatMode === CHAT_MODES.AGENTIC;
    },
    formattedDuoShortcutTooltip() {
      // Chat-specific tooltip logic moved here
      // ...
    },
  },
};
</script>
```

This refactoring provides:

- **Clear separation of concerns**: Chat navigation logic is isolated from panel navigation
- **Easier maintenance**: Changes to chat navigation don't affect sessions or suggestions
- **Better scalability**: New chat features can be added to the container without modifying the rail
- **Improved testability**: Chat navigation can be tested independently from panel navigation

### Content Container

The `content_container.vue` component serves as the rendering layer between the AI panel infrastructure and sub-applications. It handles:

**Component rendering**

- Dynamically renders the active tab's component using Vue's `<component :is>` pattern
- Supports both Vue components and static HTML strings (for error messages or unavailable states)
- Uses a computed `componentKey` that includes the chat mode to force component recreation when switching between classic and agentic chat, preventing state persistence across mode changes

**Header management**

- Displays the active tab title with support for dynamic title updates via the `change-title` event
- Provides back button functionality when `showBackButton` is true
- Includes collapse and maximize/minimize controls for panel state management

**Props passthrough**

The content container receives context props (`userId`, `projectId`, `namespaceId`, etc.) and passes them to sub-applications using `v-bind="activeTab.props"`, maintaining the agnostic pattern where the container doesn't need to know about specific sub-app requirements.

**Event relay**

Acts as an event relay between sub-applications and the AI panel:

- `@switch-to-active-tab` - Allows sub-applications to trigger navigation to other tabs
- `@change-title` - Enables sub-applications to dynamically update the panel header title
- `@closePanel` and `@toggleMaximize` - Relays panel state changes upward

**Accessibility**

- Uses proper ARIA labels for all interactive elements
- Implements tooltip directives for button actions
- Maintains semantic HTML structure with `<aside>` and heading elements

The content container is intentionally kept simple and focused on rendering concerns, delegating all business logic to sub-applications while providing consistent UI infrastructure.

### Why separate sub-applications instead of a monolithic app

The AI panel architecture deliberately maintains separate sub-applications rather than merging them into a single monolithic application.

**Benefits**

1. **Team autonomy and velocity** - Multiple teams (Duo Chat, Agent foundations, Project Studio) can deploy changes independently without coordinating releases or waiting for other teams.

1. **Reduced coupling** - Changes to one feature don't risk breaking another team's functionality, reducing the need for extensive cross-team testing.

1. **Migration and evolution** - New AI features can be added and legacy features deprecated without forcing all sub-applications to change simultaneously.

**Accepted trade-offs**

- Shared state requires explicit coordination through Apollo cache
- Some code duplication may occur across sub-applications
- Navigation between features requires event-based communication
- Initial setup complexity is higher than a monolithic approach

The panel infrastructure provides enough shared functionality (navigation, layout, state management patterns) to minimize duplication while preserving independence.

### Global state management

The AI panel already has a global state shared across all apps in the form of `vue-apollo`. Any other state like VueX, Pinia or Observables should be avoided at all cost.

**Legacy VueX store**

The `init_duo_panel.js` file currently includes a VueX store used by both classic and agentic Duo Chat implementations. This VueX store should not be expanded or used by any new features or sub-applications, and we should aim to remove it entirely from existing chat implementations. All state management must use Apollo Client's cache as described below.

**Migration plan: VueX to Apollo**

The current VueX store in Duo Chat manages message state and chat interactions. To migrate to Apollo Client's cache:

**Current VueX usage**

The VueX store currently handles:

- Storing chat messages (both sent and received)
- Managing pending message states
- Tracking conversation history

**Migration approach**

1. **Messages are already in Apollo cache** - Chat messages are fetched via GraphQL queries, meaning they already exist in the Apollo cache. We can leverage this existing cache instead of duplicating state in VueX.

1. **Move message state to component data** - The root component of `agentic_duo_chat` can hold messages as a data property and pass them as props to child components, eliminating the need for a global store for this use case. Or, child components can query the Apollo cache directly and get the latest data reactively.

1. **Handle pending messages with optimistic responses** - When sending a message, use Apollo's built-in [optimisticResponse](https://apollo.vuejs.org/api/apollo-mutation.html#optimisticresponse) feature to handle pending states:

```javascript
// In chat component
async sendMessage(content) {
  await this.$apollo.mutate({
    mutation: sendMessageMutation,
    variables: { content, conversationId: this.conversationId },
    optimisticResponse: {
      __typename: 'Mutation',
      sendMessage: {
        __typename: 'ChatMessage',
        id: `temp-${Date.now()}`,
        content,
        author: this.currentUser,
        timestamp: new Date().toISOString(),
      }
    },
    // Apollo automatically updates queries that include ChatMessage
    // and replaces the optimistic response with real data when it arrives
  });
}
```

Apollo automatically handles the lifecycle:

- Adds the optimistic message to cache immediately (instant UI feedback)
- When the real response arrives, automatically removes the optimistic entry
- Replaces it with the real message data from the server
- No manual cleanup or filtering required

**Benefits of migration**

- **Single source of truth** - Eliminates dual state management (VueX + Apollo)
- **Automatic cache updates** - Apollo's reactive cache propagates changes across all components
- **Reduced complexity** - No need to manually sync VueX store with GraphQL responses
- **Better performance** - Leverages Apollo's built-in caching and normalization
- **Consistency** - Aligns with the rest of the AI panel architecture

**Migration steps**

1. Create GraphQL type definitions for client-only pending message state
2. Implement local resolvers for pending messages in Apollo Client configuration
3. Update chat components to read from Apollo cache instead of VueX store
4. Replace VueX actions with Apollo mutations and cache updates
5. Remove VueX store initialization from `init_duo_panel.js`
6. Test message sending, receiving, and pending states work correctly
7. Verify no regressions in chat functionality across both classic and agentic modes

#### Global state management proposal

The AI panel should leverage Apollo Client's cache as the single source of truth for shared state across all sub-applications. This approach maintains consistency with our existing GraphQL infrastructure while avoiding additional state management libraries.

**Shared Apollo Provider**

The `apolloProvider` is created once during initialization in `init_duo_panel.js` and injected into the Vue instance. This ensures all sub-applications (chat, agents, sessions) share the same Apollo Client instance and cache, enabling:

- Automatic cache updates propagating across all active sub-applications
- Consistent data fetching and caching behavior
- Reduced network requests through shared cache hits

**Client-only fields with local resolvers**

For UI state that needs to be shared across sub-applications (e.g., active agent selection, panel preferences, temporary flags), we can use Apollo's `@client` directive with local resolvers.

First, create the GraphQL query file `graphql/queries/get_ai_panel_state.query.graphql`:

```graphql
query GetAiPanelState {
  aiPanelState @client {
    selectedAgent
    userPreferences
  }
}
```

Then configure the Apollo Client with type definitions and resolvers in `init_duo_panel.js`:

```javascript
import aiPanelStateTypeDefs from './graphql/typedefs/ai_panel_state.typedefs.graphql';
import getAiPanelStateQuery from './graphql/queries/get_ai_panel_state.query.graphql';

const resolvers = {
  Query: {
    aiPanelState: () => ({
      __typename: 'AiPanelState',
    }),
  },
};

const apolloProvider = new VueApollo({
  defaultClient: createDefaultClient({
    typeDefs: aiPanelStateTypeDefs,
    resolvers,
  }),
});

// Initialize cache with default values
apolloProvider.defaultClient.cache.writeQuery({
  query: getAiPanelStateQuery,
  data: {
    aiPanelState: {
      __typename: 'AiPanelState',
      selectedAgent: null,
      userPreferences: null,
    },
  },
});
```

Create the type definitions file `graphql/typedefs/ai_panel_state.typedefs.graphql`:

```graphql
extend type Query {
  aiPanelState: AiPanelState!
}

type AiPanelState {
  selectedAgent: Agent
  userPreferences: PanelPreferences
}
```

Sub-applications can then read and write this shared state using Vue Apollo's component options:

```vue
<script>
import getAiPanelStateQuery from './graphql/queries/get_ai_panel_state.query.graphql';

export default {
  name: 'AgentSelector',
  apollo: {
    aiPanelState: {
      query: getAiPanelStateQuery,
    },
  },
  computed: {
    selectedAgent() {
      return this.aiPanelState?.selectedAgent;
    },
  },
  methods: {
    selectAgent(agent) {
      this.$apollo.provider.defaultClient.cache.writeQuery({
        query: getAiPanelStateQuery,
        data: {
          aiPanelState: {
            __typename: 'AiPanelState',
            selectedAgent: agent,
          },
        },
      });
    },
  },
};
</script>

<template>
  <div>
    <p v-if="selectedAgent">Selected: {{ selectedAgent.name }}</p>
    <button @click="selectAgent(newAgent)">Select Agent</button>
  </div>
</template>
```

This pattern provides:

- Type-safe state management through GraphQL schema
- Reactive updates across all components watching the same cache fields
- Familiar GraphQL patterns for developers already using Apollo
- No additional state management dependencies
- Clear separation between server data and client-only UI state

### Permissions

The AI panel architecture separates permission concerns across three layers: Rails backend, panel infrastructure, and individual sub-applications.

**User-scoped permissions**

The AI panel is user-scoped, meaning it is present on every page across GitLab (different projects, groups, settings pages, etc.) and persists as users navigate throughout the application. Because of this global presence, all permission checks must be performed at the user level rather than being tied to specific projects or groups.

This has important implications:

- Permission checks cannot assume a specific project or group context
- Features must verify user-level entitlements (e.g., GitLab Duo Pro subscription, seat assignments)
- Context-specific permissions (e.g., project-level feature flags) should be handled by sub-applications, not the panel infrastructure
- The panel must gracefully handle transitions between different permission contexts as users navigate

**Rails layer (Backend)**

The Rails backend is responsible for:

- Determining whether the AI panel should be rendered at all (via layout/view logic)
- Providing initial configuration through data attributes on the `#duo-chat-panel` element
- Setting the `chatDisabledReason` prop when chat features are disabled at the namespace/project level
- Controlling feature availability through props like `agenticAvailable` and `agenticUnavailableMessage`

The backend passes permission-related data to the frontend through the element's dataset:

```javascript
const {
  chatDisabledReason,
  agenticAvailable,
  agenticUnavailableMessage,
  // ... other config
} = el.dataset;
```

**AI panel layer (Infrastructure)**

The panel infrastructure handles permission enforcement at the UI level:

- The `navigation_rail.vue` component receives `chatDisabledReason` and applies disabled states to navigation buttons
- When `isChatDisabled` is true, the panel prevents tab switching and displays tooltips explaining why features are unavailable
- The panel does not make permission decisions itself; it only enforces states provided by the backend

**Sub-application layer (Feature-specific)**

Individual sub-applications (Duo Chat, Agentic Chat, Sessions) are responsible for:

- Checking their own feature-specific permissions
- Handling graceful degradation when features are partially available
- Displaying appropriate error messages or fallback UI

For example, the `chatConfiguration` in `init_duo_panel.js` determines which chat component to load:

```javascript
const chatConfiguration = {
  agenticComponent: parseBoolean(agenticAvailable)
    ? DuoAgenticChat
    : agenticUnavailableMessage || __('Chat is not available.'),
  classicComponent: DuoChat,
  // ...
};
```

This layered approach ensures:

- Permission logic remains in the backend where it can be properly enforced
- The panel infrastructure stays agnostic and reusable
- Sub-applications can implement feature-specific permission handling
- Clear separation of concerns prevents permission checks from being duplicated across layers

### Context Passing

The AI panel receives page context from the Rails backend and passes it to sub-applications, enabling context-aware AI features. Currently, this context is primarily consumed by Duo Chat for providing relevant assistance based on the user's current page.

**Context data flow**

Context flows from Rails → Panel initialization → AI Panel component → Sub-applications:

1. **Rails backend** sets context data attributes on the `#duo-chat-panel` element:

```ruby
# Example: Rails view rendering the panel element
<div id="duo-chat-panel"
     data-user-id="<%= current_user.id %>"
     data-project-id="<%= @project.id %>"
     data-namespace-id="<%= @project.namespace.id %>"
     data-root-namespace-id="<%= @project.root_namespace.id %>"
     data-resource-id="<%= @resource.id %>"
     data-metadata="<%= @metadata.to_json %>">
</div>
```

1. **Panel initialization** (`init_duo_panel.js`) extracts context from the dataset:

```javascript
const {
  userId,
  projectId,
  namespaceId,
  rootNamespaceId,
  resourceId,
  metadata,
  // ...
} = el.dataset;
```

1. **AI Panel component** receives context as props and passes them to sub-applications via the props passthrough pattern:

```javascript
// In ai_panel.vue currentTabComponent computed property
return {
  component: this.currentChatComponent,
  props: {
    mode: chatMode,
    ...this.chatConfiguration.defaultProps  // Contains context props
  },
};
```

1. **Sub-applications** receive context through props defined in `chatConfiguration.defaultProps`:

```javascript
const chatConfiguration = {
  defaultProps: {
    isEmbedded: true,
    userId,
    projectId,
    namespaceId,
    rootNamespaceId,
    resourceId,
    metadata,
    // ...
  },
};
```

**Dynamic context updates**

Some context values are updated dynamically at runtime. For example, `resourceId` is overridden with the active work item ID when available:

```javascript
render(createElement) {
  const latestActiveWorkItemId = activeWorkItemIds.value[activeWorkItemIds.value.length - 1];
  return createElement(AIPanel, {
    props: {
      resourceId: latestActiveWorkItemId ?? resourceId,
      // ...
    },
  });
}
```

**Context consumption**

Currently, **Duo Chat** is the primary consumer of this context, using it to:

- Provide project-specific suggestions and assistance
- Understand the user's current workspace (project, namespace)
- Access resource-specific information (e.g., current issue, merge request)
- Tailor responses based on metadata about the current page

Other sub-applications (Sessions) may not require or use all context props, but they receive them through the passthrough pattern, allowing them to access context if needed in the future without requiring changes to the panel infrastructure.

**Benefits of this approach**

- Context is provided once at initialization and flows naturally through the component tree
- The panel remains agnostic about which context props are meaningful to which sub-applications
- New context properties can be added without modifying the panel component
- Sub-applications can selectively use only the context they need

#### Problems

The current context passing implementation has several significant limitations that impact the AI panel's ability to provide context-aware assistance across GitLab:

**Limited page coverage**

Context passing currently works reliably only on work item pages. The panel and its sub-applications lack context awareness on:

- Repository pages (no context or `resourceId` passed, though `projectId` is available during workflow creation)
- Job pages (often guesses the wrong project)
- Pipeline pages
- Most other GitLab pages beyond work items

This creates an inconsistent user experience where AI features work well in some areas but fail to understand the user's context in others.

**Static context in SPAs**

The current implementation is controller-based, meaning context is set once during page load through Rails data attributes. This creates problems for Single Page Applications (SPAs):

- Context doesn't update as users navigate within an SPA
- The panel remains unaware of route changes and state transitions
- Dynamic content changes (e.g., switching between merge requests in a list view) don't trigger context updates

**Inconsistent implementation patterns**

Different pages use different mechanisms to provide context:

- Some pages use `Gitlab::ApplicationContext.push(ai_resource_id)` in Rails controllers
- Older pages use the HAML `provide` mechanism (predates Project Studio)
- Work items use dynamic `resourceId` updates via `activeWorkItemIds` observable
- No standardized approach exists for new pages

**Missing context for key features**

The lack of comprehensive context prevents several important use cases:

- **Onboarding flows**: Cannot determine if user is on a group page without Duo enabled vs. a project with Duo
- **Suggestions tab**: Cannot provide relevant suggestions based on current page context (currently not implemented)
- **Duo Chat accuracy**: Limited ability to provide accurate responses outside of work item contexts
- **Future AI features**: Any new panel features requiring page context face the same limitations

**Ownership and coordination challenges**

- Each team is responsible for providing context for their pages
- No clear ownership of the overall context strategy
- This has been a long-standing problem (years) without a coordinated solution

**Proposed solution: Three-tiered context gathering**

The recommended approach combines multiple strategies to provide comprehensive context coverage while minimizing manual coordination (see [issue #573737](https://gitlab.com/gitlab-org/gitlab/-/issues/573737)):

**Tier 1: Initial controller data (Low effort, high value)**

Pass initial context through the Rails controller on page load via data attributes. This provides baseline context with minimal effort and is already partially implemented. Expand coverage to more entity types (`@group`, `@project`, `@merge_request`, `@pipeline`, etc.) across GitLab pages. This is already in effect.

**Tier 2: URL watching for SPA navigation (Medium effort, AI team owned)**

Watch `window.location` for changes to detect in-app navigation. When the URL changes:

1. Check if the main entity context remains the same (e.g., still within the same project)
1. Parse the URL path to extract granular context using regex patterns:

```javascript
const contextPatterns = [
  {
    pattern: /\/merge_requests\/(\d+)/,
    extract: (match) => ({ type: 'merge_request', iid: match[1] })
  },
  {
    pattern: /\/issues\/(\d+)/,
    extract: (match) => ({ type: 'issue', iid: match[1] })
  },
  {
    pattern: /\/pipelines\/(\d+)/,
    extract: (match) => ({ type: 'pipeline', id: match[1] })
  },
  {
    pattern: /\/blob\/([^\/]+)\/(.+)/,
    extract: (match) => ({ type: 'file', ref: match[1], path: match[2] })
  }
  // Add patterns incrementally for high-value routes
];
```

1. If the main context changed (e.g., navigating from pipelines to work items), fire a GraphQL query to fetch the new entity data and update the Apollo cache. This will be rare given out cluster-spa approach, but it could happen in the the future.
1. Update the AI panel context through Apollo cache mutations

This approach allows the AI team to own the URL pattern matching and incrementally expand coverage without requiring coordination with other teams.

**Tier 3: Browser events for cross-app communication (Targeted effort)**

Use native browser `CustomEvent` to handle edge cases where context changes don't follow standard routing patterns. This is necessary because GitLab has multiple independent Vue apps on the same page (AI panel, work item drawer, main app, etc.), each with their own Vue instance and Apollo provider that don't share the same Apollo cache instance. Since the drawer is present on every app, it is also non-trivial to always have it share the same Apollo cache as other apps on the page.

Examples of edge cases include:

- Work item drawer opening/closing (updates `window.location` with query params but isn't router-controlled)
- Modal dialogs with contextual information
- Dynamic content loaded without URL changes

Any Vue app can dispatch context updates via browser events:

```javascript
// In work item drawer wrapper (or any other Vue app)
window.dispatchEvent(new CustomEvent('ai-panel:context-update', {
  detail: {
    resourceId: workItemId,
    resourceType: 'work_item',
    subContext: 'drawer_view',
  }
}));
```

The AI panel listens for these events and updates its own Apollo cache:

```javascript
// In AI panel initialization (init_duo_panel.js)
window.addEventListener('ai-panel:context-update', (event) => {
  const context = event.detail;

  // Update AI panel's Apollo cache
  apolloProvider.defaultClient.cache.writeQuery({
    query: getAiPanelContextQuery,
    data: {
      aiPanelContext: {
        __typename: 'AiPanelContext',
        ...getCurrentContext(), // Preserve existing context
        ...context, // Merge in new context
      },
    },
  });
});
```

This approach works across all Vue apps because:

- Browser events are global and not scoped to Vue instances
- No shared dependencies or imports required between apps
- Simple to implement and debug with clear event naming conventions
- Can pass structured data via the `detail` property
- Easy to debug by ensuring the right browser events are emitted when expected

### Testing and CI

For information about testing infrastructure and CI/CD configuration related to this work:

- **Main epic**: [Project Studio switchover](https://gitlab.com/groups/gitlab-org/-/epics/19140)
- **Switchover timeline**: [Timeline and tasks](https://gitlab.com/gitlab-org/gitlab/-/work_items/578461) (maintained as the source of truth)

**Future maintenance tasks**:

When the Project Studio feature flag is removed:

- Remove classic UI jobs from CI configuration
- Remove ProjectStudio conditionals from test files

When the Project Studio banner is removed:

- Remove test code handling the banner display

Details for test/CI code removal in the above tasks (some of these will be overlapping or mutually dependent):

1. Remove CI jobs `rspec system pg16 classic-ui` and `rspec-ee system pg16 classic-ui`
1. Remove `GLCI_OVERRIDE_PROJECT_STUDIO_ENABLED` and all references/usages
1. Remove `Users::ProjectStudio.enabled_for_user?` and all references/usages
1. Remove entire `Users::ProjectStudio` module and all references/usages
1. Remove `paneled_view` feature flag and all references/usages
